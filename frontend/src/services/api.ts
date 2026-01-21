import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export type SourceType = 'dayone' | 'wordpress';

export interface SearchParams {
  query: string;
  limit?: number;
  source?: SourceType;
}

export const search = async ({ query, limit = 5, source }: SearchParams) => {
  const response = await api.get('/search', {
    params: { q: query, limit, source },
  });
  return response.data;
};
