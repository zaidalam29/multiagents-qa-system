import json
import os
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
from functools import lru_cache
from app.config import settings

class VectorStore:
    def __init__(self):
        self.db_path = os.path.join(settings.VECTOR_DB_PATH, "vectors.db")
        self.embedding_cache = {}  # Simple in-memory cache
        self._init_database()
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get cached embedding or compute new one"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Compute embedding
        embeddings = settings.get_embeddings([text])
        embedding = embeddings[0]
        
        # Cache it
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    def _init_database(self):
        """Initialize SQLite database with migration support"""
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Check if database exists and needs migration
        db_exists = os.path.exists(self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if db_exists:
            # Check current schema
            try:
                cursor.execute("SELECT collection_name FROM documents LIMIT 1")
                # Column exists, no migration needed
                conn.close()
                return
            except sqlite3.OperationalError:
                # Column doesn't exist, need to migrate
                print("Migrating database to new schema...")
                self._migrate_database(conn, cursor)
        else:
            # Create fresh database
            self._create_fresh_database(conn, cursor)
        
        conn.close()
    
    def _create_fresh_database(self, conn, cursor):
        """Create fresh database with new schema"""
        # Drop old tables if exist
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("DROP TABLE IF EXISTS collections")
        
        # Create collections table
        cursor.execute('''
            CREATE TABLE collections (
                name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create documents table with collection_name
        cursor.execute('''
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB NOT NULL,
                collection_name TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_name) REFERENCES collections(name)
            )
        ''')
        
        # Create index
        cursor.execute('''
            CREATE INDEX idx_documents_collection 
            ON documents(collection_name)
        ''')
        
        conn.commit()
        print("Created fresh database with new schema")
    
    def _migrate_database(self, conn, cursor):
        """Migrate old database to new schema"""
        # Get all data from old table
        cursor.execute("SELECT id, content, metadata, embedding FROM documents")
        old_data = cursor.fetchall()
        
        # Drop old table
        cursor.execute("DROP TABLE IF EXISTS documents")
        
        # Create new tables
        cursor.execute('''
            CREATE TABLE collections (
                name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB NOT NULL,
                collection_name TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_name) REFERENCES collections(name)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX idx_documents_collection 
            ON documents(collection_name)
        ''')
        
        # Ensure default collection exists
        cursor.execute(
            "INSERT OR IGNORE INTO collections (name) VALUES (?)",
            ('default',)
        )
        
        # Reinsert data with default collection
        for row in old_data:
            doc_id, content, metadata, embedding = row
            
            # Parse metadata and add collection
            try:
                meta_dict = json.loads(metadata)
                meta_dict['collection'] = 'default'
                new_metadata = json.dumps(meta_dict)
            except:
                new_metadata = metadata
            
            cursor.execute(
                """INSERT INTO documents 
                   (id, content, metadata, embedding, collection_name) 
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, content, new_metadata, embedding, 'default')
            )
        
        conn.commit()
        print(f"Migrated {len(old_data)} documents to new schema")
    
    def _generate_id(self, text: str, metadata: Dict) -> str:
        """Generate unique ID for document"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        meta_str = json.dumps(metadata, sort_keys=True)
        meta_hash = hashlib.md5(meta_str.encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{meta_hash}"
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                     collection_name: str = "default") -> List[str]:
        """Add documents to vector store with collection support"""
        if len(documents) != len(metadatas):
            raise ValueError("Documents and metadatas must have same length")
        
        # Get embeddings in batch
        embeddings = settings.get_embeddings(documents)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure collection exists
        cursor.execute(
            "INSERT OR IGNORE INTO collections (name) VALUES (?)",
            (collection_name,)
        )
        
        ids = []
        for doc, meta, emb in zip(documents, metadatas, embeddings):
            doc_id = self._generate_id(doc, meta)
            
            # Add collection to metadata
            meta_with_collection = meta.copy()
            meta_with_collection['collection'] = collection_name
            
            # Convert embedding to bytes
            emb_array = np.array(emb, dtype=np.float32)
            emb_bytes = emb_array.tobytes()
            
            # Insert or replace document
            cursor.execute(
                """INSERT OR REPLACE INTO documents 
                   (id, content, metadata, embedding, collection_name) 
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, doc, json.dumps(meta_with_collection), emb_bytes, collection_name)
            )
            
            ids.append(doc_id)
        
        conn.commit()
        conn.close()
        return ids
    
    def search(self, query: str, top_k: int = 5, collection_name: Optional[str] = None) -> List[Dict]:
        """Search for similar documents with cached embeddings"""
        # Get cached query embedding
        query_embedding = self._get_cached_embedding(query)
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query based on collection
        if collection_name:
            cursor.execute(
                "SELECT id, content, metadata, embedding FROM documents WHERE collection_name = ?",
                (collection_name,)
            )
        else:
            cursor.execute("SELECT id, content, metadata, embedding FROM documents")
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            doc_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            
            # Calculate cosine similarity
            norm_query = np.linalg.norm(query_vec)
            norm_doc = np.linalg.norm(doc_embedding)
            
            if norm_query > 0 and norm_doc > 0:
                similarity = np.dot(query_vec, doc_embedding) / (norm_query * norm_doc)
            else:
                similarity = 0
            
            metadata = json.loads(row['metadata'])
            
            results.append({
                "id": row['id'],
                "document": row['content'],
                "metadata": metadata,
                "score": float(similarity),
                "collection": metadata.get('collection', 'default')
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_collection_stats(self, collection_name: str = "default") -> Dict:
        """Get statistics for a collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get document count
        cursor.execute(
            "SELECT COUNT(*) FROM documents WHERE collection_name = ?",
            (collection_name,)
        )
        count = cursor.fetchone()[0]
        
        # Try to get unique sources
        try:
            cursor.execute(
                """SELECT COUNT(DISTINCT json_extract(metadata, '$.source')) 
                   FROM documents WHERE collection_name = ?""",
                (collection_name,)
            )
            sources = cursor.fetchone()[0]
        except:
            sources = 0
        
        conn.close()
        
        return {
            "collection": collection_name,
            "document_count": count,
            "unique_sources": sources
        }
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM collections ORDER BY created_at")
        collections = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return collections
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self._get_cached_embedding.cache_clear()