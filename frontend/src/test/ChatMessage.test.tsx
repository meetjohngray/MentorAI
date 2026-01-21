import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessage } from '../components/ChatMessage';
import type { ChatMessage as ChatMessageType, Source } from '../types';

describe('ChatMessage', () => {
  it('renders user message correctly', () => {
    const message: ChatMessageType = {
      role: 'user',
      content: 'Hello, mentor!',
    };

    render(<ChatMessage message={message} />);

    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Hello, mentor!')).toBeInTheDocument();
  });

  it('renders assistant message correctly', () => {
    const message: ChatMessageType = {
      role: 'assistant',
      content: 'Hello! How can I help you reflect today?',
    };

    render(<ChatMessage message={message} />);

    expect(screen.getByText('Mentor')).toBeInTheDocument();
    expect(
      screen.getByText('Hello! How can I help you reflect today?')
    ).toBeInTheDocument();
  });

  it('renders markdown in assistant messages', () => {
    const message: ChatMessageType = {
      role: 'assistant',
      content: 'Here is a **bold** statement.',
    };

    render(<ChatMessage message={message} />);

    // The bold text should be wrapped in a <strong> element
    const boldElement = screen.getByText('bold');
    expect(boldElement.tagName).toBe('STRONG');
  });

  it('does not show sources panel for user messages', () => {
    const message: ChatMessageType = {
      role: 'user',
      content: 'Test message',
    };
    const sources: Source[] = [
      {
        id: 'src-1',
        text: 'Source text',
        source_type: 'dayone',
        relevance_score: 0.9,
      },
    ];

    render(<ChatMessage message={message} sources={sources} />);

    expect(screen.queryByText(/source/i)).not.toBeInTheDocument();
  });

  it('shows sources panel for assistant messages with sources', () => {
    const message: ChatMessageType = {
      role: 'assistant',
      content: 'Based on your journal...',
    };
    const sources: Source[] = [
      {
        id: 'src-1',
        text: 'Journal entry about meditation',
        source_type: 'dayone',
        date: '2024-01-15',
        relevance_score: 0.9,
      },
    ];

    render(<ChatMessage message={message} sources={sources} />);

    expect(screen.getByText('1 source referenced')).toBeInTheDocument();
  });

  it('does not show sources panel when sources array is empty', () => {
    const message: ChatMessageType = {
      role: 'assistant',
      content: 'A general response',
    };

    render(<ChatMessage message={message} sources={[]} />);

    expect(screen.queryByText(/source/i)).not.toBeInTheDocument();
  });
});
