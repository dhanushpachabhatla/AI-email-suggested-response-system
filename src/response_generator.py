"""
Response Generator for AI Email Response System.

Implements RAG-based email response generation using:
- sentence-transformers for embeddings (all-MiniLM-L6-v2)
- ChromaDB for vector storage and similarity search
- LM Studio (primary) or cloud APIs (fallback) for text generation
"""

import time
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.dataset_manager import EmailPair
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class GeneratedResponse:
    """Result of a single email response generation."""
    incoming_email: str
    generated_text: str
    retrieved_examples: List[EmailPair]
    retrieval_scores: List[float]
    retrieval_quality: float       # Average similarity of retrieved examples
    generation_latency_ms: float
    prompt_tokens_estimate: int    # Rough estimate


# ---------------------------------------------------------------------------
# Embedding Manager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """
    Manages text embeddings using sentence-transformers.

    Uses all-MiniLM-L6-v2 (384-dim) — fast inference with good
    semantic similarity performance for sentence-level tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: Hugging Face model identifier.
                        Alternative: 'all-mpnet-base-v2' (768-dim, higher quality, slower).

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded: {model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. "
                f"Ensure sentence-transformers is installed and the model name is correct. "
                f"Error: {e}"
            ) from e

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed. Empty list returns empty array.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.debug(f"Generated embeddings for {len(texts)} text(s)")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method to embed a single text string."""
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Manages ChromaDB vector storage for email-response pairs.

    Persists embeddings to disk for reuse across sessions.
    Falls back to in-memory mode on persistence failures.
    """

    def __init__(
        self,
        collection_name: str = "email_embeddings",
        persistence_dir: str = "./data/embeddings",
    ):
        """
        Initialize ChromaDB client and collection.

        Args:
            collection_name: Name for the ChromaDB collection.
            persistence_dir: Directory for persistent storage.
        """
        self.collection_name = collection_name
        self.persistence_dir = persistence_dir

        try:
            self.client = chromadb.PersistentClient(path=persistence_dir)
            logger.info(f"ChromaDB persistent client initialised at '{persistence_dir}'")
        except Exception as e:
            logger.warning(
                f"Could not initialise persistent ChromaDB client ({e}). "
                "Falling back to in-memory mode."
            )
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store collection '{collection_name}' ready "
                    f"({self.collection.count()} existing documents)")

    def add_email_pairs(
        self,
        pairs: List[EmailPair],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add email pairs with their embeddings to the vector store.

        Args:
            pairs: List of EmailPair objects.
            embeddings: numpy array of shape (len(pairs), dim).
        """
        if not pairs:
            logger.warning("add_email_pairs called with empty list — skipping.")
            return

        ids = [p.id for p in pairs]
        documents = [p.incoming_email for p in pairs]
        metadatas = [
            {
                "response": p.response,
                "subject": p.metadata.subject,
                "formality_level": p.metadata.formality_level,
                "email_type": p.metadata.email_type,
                "subject_category": p.metadata.subject_category,
                "sender_role": p.metadata.sender_role,
            }
            for p in pairs
        ]

        # ChromaDB expects plain Python lists, not numpy arrays
        embedding_list = embeddings.tolist()

        # Upsert to handle re-indexing without duplicate errors
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )
        logger.info(f"Added/updated {len(pairs)} email pairs in vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> tuple[List[EmailPair], List[float]]:
        """
        Retrieve top-k most similar email pairs using cosine similarity.

        Args:
            query_embedding: 1-D numpy embedding of the query.
            top_k: Number of results to return.

        Returns:
            Tuple of (list_of_EmailPair, list_of_similarity_scores)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        email_pairs: List[EmailPair] = []
        scores: List[float] = []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = 1.0 - (dist / 2.0)  # Convert distance → similarity

            from src.dataset_manager import EmailMetadata
            ep = EmailPair(
                id="",
                incoming_email=doc,
                response=meta["response"],
                metadata=EmailMetadata(
                    subject=meta.get("subject", ""),
                    formality_level=meta.get("formality_level", "semi-formal"),
                    email_type=meta.get("email_type", "professional"),
                    subject_category=meta.get("subject_category", "inquiry"),
                    sender_role=meta.get("sender_role", "unknown"),
                ),
            )
            email_pairs.append(ep)
            scores.append(round(similarity, 4))

        logger.debug(f"Retrieved {len(email_pairs)} similar pairs (top_k={top_k})")
        return email_pairs, scores

    def count(self) -> int:
        """Return the number of stored documents."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all items from the collection (useful for re-indexing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' cleared")


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM interface: LM Studio (primary) with optional cloud fallback.

    Primary: Local LM Studio server (OpenAI-compatible API, port 1234).
    Fallback: OpenAI or Anthropic APIs when LM Studio is unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dict with keys:
                - lm_studio_url (str): Base URL, e.g. "http://127.0.0.1:1234/v1"
                - timeout (int): Request timeout in seconds (default 30)
                - fallback_provider (str|None): "openai", "anthropic", or None
                - fallback_api_key (str|None): API key for fallback provider
        """
        self.lm_studio_url = config.get("lm_studio_url", "http://127.0.0.1:1234/v1").rstrip("/")
        self.timeout = config.get("timeout", 30)
        self.fallback_provider = config.get("fallback_provider", None)
        self.fallback_api_key = config.get("fallback_api_key", None)
        logger.info(f"LLMClient initialised (primary=lm_studio, fallback={self.fallback_provider})")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text. Tries LM Studio first; falls back to cloud API on failure.

        Args:
            prompt: The full prompt string.
            max_tokens: Maximum tokens in the completion.
            temperature: Sampling temperature (0.0–1.0).

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If all providers fail and no fallback is configured.
        """
        try:
            return self._call_lm_studio(prompt, max_tokens, temperature)
        except Exception as primary_err:
            logger.warning(f"LM Studio call failed: {primary_err}")
            if self.fallback_provider:
                logger.info(f"Attempting fallback via {self.fallback_provider}")
                return self._call_fallback(prompt, max_tokens, temperature)
            raise RuntimeError(
                f"LM Studio not reachable and no fallback configured. "
                f"Ensure LM Studio is running at {self.lm_studio_url}. "
                f"Error: {primary_err}"
            ) from primary_err

    def _call_lm_studio(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call local LM Studio using OpenAI-compatible chat completions endpoint."""
        url = f"{self.lm_studio_url}/chat/completions"
        payload = {
            "model": "local-model",   # LM Studio ignores this; uses the loaded model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"LM Studio not reachable at {self.lm_studio_url}. "
                "Please start LM Studio and load a model."
            ) from e
        except requests.exceptions.Timeout:
            logger.error(f"LM Studio request timed out after {self.timeout}s")
            raise TimeoutError(f"LM Studio request timed out after {self.timeout}s")

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected LM Studio response format: {data}") from e

        if not content or not content.strip():
            logger.warning("LM Studio returned an empty response")
            return ""

        return content.strip()

    def _call_fallback(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call cloud API fallback with exponential backoff for rate limits."""
        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.fallback_provider == "openai":
                    return self._call_openai(prompt, max_tokens, temperature)
                elif self.fallback_provider == "anthropic":
                    return self._call_anthropic(prompt, max_tokens, temperature)
                else:
                    raise ValueError(f"Unknown fallback provider: {self.fallback_provider}")
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt+1})")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    logger.error(f"Fallback provider '{self.fallback_provider}' failed: {e}")
                    raise
                else:
                    raise

        raise RuntimeError(f"Fallback provider '{self.fallback_provider}' exhausted retries")

    def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI chat completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.fallback_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Anthropic Messages API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.fallback_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    def check_connectivity(self) -> bool:
        """
        Check if LM Studio is reachable.

        Returns:
            True if LM Studio responds, False otherwise.
        """
        try:
            resp = requests.get(f"{self.lm_studio_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Response Generator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Orchestrates RAG-based email response generation.

    Flow: Embed query → Retrieve similar pairs → Build few-shot prompt → Call LLM → Return response
    """

    def __init__(
        self,
        embedding_mgr: EmbeddingManager,
        vector_store: VectorStore,
        llm_client: LLMClient,
    ):
        """
        Args:
            embedding_mgr: EmbeddingManager instance for text→vector conversion.
            vector_store: VectorStore instance for similarity retrieval.
            llm_client: LLMClient instance for text generation.
        """
        self.embedding_mgr = embedding_mgr
        self.vector_store = vector_store
        self.llm_client = llm_client
        logger.info("ResponseGenerator initialised")

    def generate_response(
        self,
        incoming_email: str,
        top_k: int = 3,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """
        Generate an email response using RAG.

        Args:
            incoming_email: The email text to respond to.
            top_k: Number of similar examples to retrieve.
            max_tokens: Max tokens for LLM generation.
            temperature: LLM sampling temperature.

        Returns:
            GeneratedResponse with response text, retrieved examples, and metadata.
        """
        start_time = time.time()

        # Step 1: Embed the incoming email
        query_embedding = self.embedding_mgr.embed_single(incoming_email)

        # Step 2: Retrieve top-k similar email pairs
        retrieved_pairs, retrieval_scores = self.vector_store.search(query_embedding, top_k=top_k)

        # Compute retrieval quality (average similarity)
        retrieval_quality = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        # Step 3: Build few-shot prompt
        prompt = self.build_few_shot_prompt(incoming_email, retrieved_pairs)

        # Step 4: Call LLM
        generated_text = self.llm_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        if not generated_text.strip():
            logger.warning("LLM returned an empty response")
            generated_text = "[No response generated]"

        # Calculate latency
        elapsed_ms = (time.time() - start_time) * 1000

        # Estimate prompt tokens (rough heuristic: 1 token ≈ 4 characters)
        prompt_tokens_estimate = len(prompt) // 4

        logger.debug(
            f"Generated response in {elapsed_ms:.1f}ms (retrieval_quality={retrieval_quality:.3f})"
        )

        return GeneratedResponse(
            incoming_email=incoming_email,
            generated_text=generated_text,
            retrieved_examples=retrieved_pairs,
            retrieval_scores=retrieval_scores,
            retrieval_quality=round(retrieval_quality, 4),
            generation_latency_ms=round(elapsed_ms, 2),
            prompt_tokens_estimate=prompt_tokens_estimate,
        )

    def build_few_shot_prompt(self, incoming: str, examples: List[EmailPair]) -> str:
        """
        Construct a few-shot prompt with retrieved examples.

        Format:
            System message
            Example 1: Incoming → Response
            Example 2: Incoming → Response
            ...
            Now respond to this email: {incoming}

        Args:
            incoming: The email to respond to.
            examples: List of EmailPair examples (retrieved from vector store).

        Returns:
            Full prompt string.
        """
        system_msg = (
            "You are a professional email response assistant. "
            "Generate a clear, appropriate, and helpful reply to the incoming email. "
            "Match the tone and formality level of the incoming message. "
            "Be concise, address all points raised, and maintain a professional demeanor.\n\n"
        )

        prompt = system_msg
        prompt += "Here are some example email-response pairs:\n\n"

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Incoming: {example.incoming_email[:300]}...\n"  # Truncate if too long
            prompt += f"Response: {example.response[:300]}...\n\n"

        prompt += f"Now generate a professional response for this email:\n{incoming}\n\nResponse:"

        return prompt
