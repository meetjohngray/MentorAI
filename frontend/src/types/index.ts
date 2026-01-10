export interface SearchResult {
  id: string;
  text: string;
  metadata: {
    source_type: string;
    entry_id: string;
    date: string;
    tags: string;
    has_photos: boolean;
    photo_count: number;
    chunk_index: number;
    total_chunks: number;
  };
  distance: number;
  relevance_score: number;
}

export interface SearchResponse {
  query: string;
  num_results: number;
  results: SearchResult[];
}

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
