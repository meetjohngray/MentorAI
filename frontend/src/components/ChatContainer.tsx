import { useState, useRef, useEffect, useCallback } from 'react';
import type { ChatMessage as ChatMessageType, Source } from '../types';
import { sendChatMessage, getConversation } from '../services/api';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import styles from './ChatContainer.module.css';

interface MessageWithSources {
  message: ChatMessageType;
  sources?: Source[];
}

interface ChatContainerProps {
  conversationId: string | null;
  onConversationChange: (id: string) => void;
}

export function ChatContainer({
  conversationId,
  onConversationChange,
}: ChatContainerProps) {
  const [messages, setMessages] = useState<MessageWithSources[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentConversationId, setCurrentConversationId] = useState<
    string | null
  >(conversationId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load conversation when conversationId prop changes
  const loadConversation = useCallback(async (id: string) => {
    try {
      const conversation = await getConversation(id);
      const loaded: MessageWithSources[] = conversation.messages.map((m) => ({
        message: { role: m.role, content: m.content },
        sources: m.sources ?? undefined,
      }));
      setMessages(loaded);
      setCurrentConversationId(id);
      setError(null);
    } catch {
      setError('Failed to load conversation');
    }
  }, []);

  useEffect(() => {
    if (conversationId) {
      // Load from API only if the prop points to a different conversation
      if (conversationId !== currentConversationId) {
        loadConversation(conversationId);
      }
    } else if (currentConversationId !== null) {
      // Parent explicitly reset to null — start fresh
      setMessages([]);
      setCurrentConversationId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, loadConversation]);

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
        conversationId: currentConversationId ?? undefined,
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

      // Update conversation ID if this was a new conversation
      if (!currentConversationId) {
        setCurrentConversationId(response.conversation_id);
        onConversationChange(response.conversation_id);
      }
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
