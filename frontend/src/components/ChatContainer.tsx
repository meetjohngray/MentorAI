import { useState, useRef, useEffect } from 'react';
import type { ChatMessage as ChatMessageType, Source } from '../types';
import { sendChatMessage } from '../services/api';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import styles from './ChatContainer.module.css';

interface MessageWithSources {
  message: ChatMessageType;
  sources?: Source[];
}

export function ChatContainer() {
  const [messages, setMessages] = useState<MessageWithSources[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    const userMessage: ChatMessageType = { role: 'user', content };

    // Add user message to the list
    setMessages((prev) => [...prev, { message: userMessage }]);
    setError(null);
    setIsLoading(true);

    try {
      // Build conversation history from existing messages
      const conversationHistory = messages.map((m) => m.message);

      const response = await sendChatMessage({
        message: content,
        conversationHistory,
      });

      const assistantMessage: ChatMessageType = {
        role: 'assistant',
        content: response.response,
      };

      setMessages((prev) => [
        ...prev,
        { message: assistantMessage, sources: response.sources },
      ]);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      // Remove the user message on error so they can retry
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.messages}>
        {messages.length === 0 && !isLoading && (
          <div className={styles.welcome}>
            <h2>Welcome</h2>
            <p>
              I'm here to help you reflect on your experiences and patterns.
              Ask me anything about what you've written in your journals or blog.
            </p>
            <p className={styles.hint}>
              Try asking about recurring themes, patterns in your thinking,
              or moments that have shaped you.
            </p>
          </div>
        )}

        {messages.map((item, index) => (
          <ChatMessage
            key={index}
            message={item.message}
            sources={item.sources}
          />
        ))}

        {isLoading && (
          <div className={styles.loading}>
            <div className={styles.loadingDots}>
              <span />
              <span />
              <span />
            </div>
            <span className={styles.loadingText}>Reflecting...</span>
          </div>
        )}

        {error && (
          <div className={styles.error}>
            <p>{error}</p>
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <ChatInput onSubmit={handleSendMessage} disabled={isLoading} />
    </div>
  );
}
