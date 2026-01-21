import Markdown from 'react-markdown';
import type { ChatMessage as ChatMessageType, Source } from '../types';
import { SourcesPanel } from './SourcesPanel';
import styles from './ChatMessage.module.css';

interface ChatMessageProps {
  message: ChatMessageType;
  sources?: Source[];
}

export function ChatMessage({ message, sources }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
      <div className={styles.role}>{isUser ? 'You' : 'Mentor'}</div>
      <div className={styles.content}>
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <Markdown>{message.content}</Markdown>
        )}
      </div>
      {!isUser && sources && sources.length > 0 && (
        <SourcesPanel sources={sources} />
      )}
    </div>
  );
}
