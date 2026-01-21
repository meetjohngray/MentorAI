// ============================================================================
// Chat Types
// ============================================================================

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  conversation_history: ChatMessage[];
}

export interface Source {
  id: string;
  text: string;
  source_type: string;
  date?: string;
  title?: string;
  relevance_score: number;
}

export interface ChatResponse {
  response: string;
  sources: Source[];
}

// ============================================================================
// Search Types
// ============================================================================

// Base metadata shared by all sources
interface BaseMetadata {
  source_type: 'dayone' | 'wordpress';
  date: string;
  tags: string;
  chunk_index: number;
  total_chunks: number;
}

// DayOne-specific metadata
interface DayOneMetadata extends BaseMetadata {
  source_type: 'dayone';
  entry_id: string;
  entry_index: number;
  has_photos: boolean;
  photo_count: number;
}

// WordPress-specific metadata
interface WordPressMetadata extends BaseMetadata {
  source_type: 'wordpress';
  post_id: string;
  title: string;
  post_index: number;
  categories: string;
}

export type SearchResultMetadata = DayOneMetadata | WordPressMetadata;

export interface SearchResult {
  id: string;
  text: string;
  metadata: SearchResultMetadata;
  distance: number;
  relevance_score: number;
}

export interface SearchResponse {
  query: string;
  num_results: number;
  results: SearchResult[];
}

// ============================================================================
// Health Types
// ============================================================================

export interface HealthResponse {
  status: string;
  version: string;
  components: {
    api: string;
    database: string;
    vector_store: string;
  };
  vector_store_documents: number;
}
