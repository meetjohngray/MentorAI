import axios from 'axios';
import type {
  ChatMessage,
  ChatResponse,
  HealthResponse,
  SearchResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Health
// ============================================================================

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

// ============================================================================
// Search
// ============================================================================

export type SourceType = 'dayone' | 'wordpress';

export interface SearchParams {
  query: string;
  limit?: number;
  source?: SourceType;
}

export const search = async ({
  query,
  limit = 5,
  source,
}: SearchParams): Promise<SearchResponse> => {
  const response = await api.get<SearchResponse>('/search', {
    params: { q: query, limit, source },
  });
  return response.data;
};

// ============================================================================
// Chat
// ============================================================================

export interface ChatParams {
  message: string;
  conversationHistory?: ChatMessage[];
}

export const sendChatMessage = async ({
  message,
  conversationHistory = [],
}: ChatParams): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/chat', {
    message,
    conversation_history: conversationHistory,
  });
  return response.data;
};
