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
  source_type: 'dayone' | 'wordpress' | 'wisdom' | 'commonplace';
  date?: string;
  title?: string;
  relevance_score: number;
  // Wisdom-specific fields
  tradition?: string;
  teacher?: string;
  text_title?: string;
  // Commonplace-specific fields
  author?: string;
  book_title?: string;
}

export interface ChatResponse {
  response: string;
  sources: Source[];
  conversation_id: string;
}

// ============================================================================
// Conversation Types
// ============================================================================

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview?: string;
}

export interface ChatMessageWithSources {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageWithSources[];
}

// ============================================================================
// Search Types
// ============================================================================

// Base metadata shared by all sources
interface BaseMetadata {
  source_type: 'dayone' | 'wordpress' | 'wisdom' | 'commonplace';
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

// Wisdom-specific metadata
interface WisdomMetadata extends BaseMetadata {
  source_type: 'wisdom';
  tradition: string;
  teacher: string;
  text_title: string;
}

// Commonplace-specific metadata
interface CommonplaceMetadata extends BaseMetadata {
  source_type: 'commonplace';
  entry_id: string;
  entry_index: number;
  author?: string;
  book_title?: string;
}

export type SearchResultMetadata = DayOneMetadata | WordPressMetadata | WisdomMetadata | CommonplaceMetadata;

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
