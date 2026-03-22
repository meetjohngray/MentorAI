import { useState, useCallback } from 'react';
import { ChatContainer } from '../components/ChatContainer';
import { ConversationSidebar } from '../components/ConversationSidebar';
import styles from './ChatPage.module.css';

export function ChatPage() {
  const [selectedConversationId, setSelectedConversationId] = useState<
    string | null
  >(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleSelectConversation = useCallback((id: string) => {
    setSelectedConversationId(id);
  }, []);

  const handleNewConversation = useCallback(() => {
    setSelectedConversationId(null);
  }, []);

  const handleConversationChange = useCallback((id: string) => {
    setSelectedConversationId(id);
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return (
    <div className={styles.page}>
      <ConversationSidebar
        activeConversationId={selectedConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        refreshTrigger={refreshTrigger}
      />
      <div className={styles.chatArea}>
        <header className={styles.header}>
          <h1 className={styles.title}>MentorAI</h1>
        </header>
        <main className={styles.main}>
          <ChatContainer
            conversationId={selectedConversationId}
            onConversationChange={handleConversationChange}
          />
        </main>
      </div>
    </div>
  );
}
