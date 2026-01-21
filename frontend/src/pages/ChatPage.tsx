import { ChatContainer } from '../components/ChatContainer';
import styles from './ChatPage.module.css';

export function ChatPage() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>MentorAI</h1>
      </header>
      <main className={styles.main}>
        <ChatContainer />
      </main>
    </div>
  );
}
