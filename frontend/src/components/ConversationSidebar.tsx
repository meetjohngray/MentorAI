import { useState, useEffect } from 'react';
import type { ConversationSummary } from '../types';
import { getConversations, deleteConversation } from '../services/api';
import styles from './ConversationSidebar.module.css';

interface ConversationSidebarProps {
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  refreshTrigger: number;
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export function ConversationSidebar({
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  refreshTrigger,
}: ConversationSidebarProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getConversations()
      .then((data) => {
        if (!cancelled) setConversations(data);
      })
      .catch(() => {
        // Silently fail — sidebar is non-critical
      });
    return () => {
      cancelled = true;
    };
  }, [refreshTrigger]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this conversation?')) return;

    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        onNewConversation();
      }
    } catch {
      // Silently fail
    }
  };

  const handleSelect = (id: string) => {
    onSelectConversation(id);
    setIsOpen(false);
  };

  const handleNew = () => {
    onNewConversation();
    setIsOpen(false);
  };

  return (
    <>
      <button
        className={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle conversations"
      >
        &#9776;
      </button>

      {isOpen && (
        <div className={styles.overlay} onClick={() => setIsOpen(false)} />
      )}

      <aside
        className={`${styles.sidebar} ${isOpen ? styles.sidebarOpen : ''}`}
      >
        <div className={styles.header}>
          <span className={styles.headerTitle}>Conversations</span>
          <button className={styles.newButton} onClick={handleNew}>
            + New
          </button>
        </div>

        <div className={styles.list}>
          {conversations.length === 0 ? (
            <div className={styles.empty}>
              No conversations yet. Start one below.
            </div>
          ) : (
            conversations.map((conv) => (
              <button
                key={conv.id}
                className={`${styles.conversation} ${
                  activeConversationId === conv.id ? styles.active : ''
                }`}
                onClick={() => handleSelect(conv.id)}
              >
                <div className={styles.conversationInfo}>
                  <div className={styles.conversationTitle}>{conv.title}</div>
                  <div className={styles.conversationMeta}>
                    {formatRelativeTime(conv.updated_at)} &middot;{' '}
                    {conv.message_count} msgs
                  </div>
                </div>
                <button
                  className={styles.deleteButton}
                  onClick={(e) => handleDelete(e, conv.id)}
                  aria-label="Delete conversation"
                >
                  &#x2715;
                </button>
              </button>
            ))
          )}
        </div>
      </aside>
    </>
  );
}
