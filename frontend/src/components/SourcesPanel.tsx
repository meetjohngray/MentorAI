import { useState } from 'react';
import type { Source } from '../types';
import styles from './SourcesPanel.module.css';

interface SourcesPanelProps {
  sources: Source[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (sources.length === 0) {
    return null;
  }

  return (
    <div className={styles.panel}>
      <button
        className={styles.toggle}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <ChevronIcon expanded={isExpanded} />
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''} referenced</span>
      </button>

      {isExpanded && (
        <div className={styles.sourceList}>
          {sources.map((source) => (
            <SourceCard key={source.id} source={source} />
          ))}
        </div>
      )}
    </div>
  );
}

function SourceCard({ source }: { source: Source }) {
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return null;
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  const sourceLabel = source.source_type === 'dayone' ? 'Journal' : 'Blog';

  return (
    <div className={styles.source}>
      <div className={styles.sourceMeta}>
        <span className={`${styles.sourceType} ${styles[source.source_type]}`}>
          {sourceLabel}
        </span>
        {source.date && (
          <span className={styles.sourceDate}>{formatDate(source.date)}</span>
        )}
        {source.title && (
          <span className={styles.sourceTitle}>{source.title}</span>
        )}
      </div>
      <p className={styles.sourceText}>{source.text}</p>
    </div>
  );
}

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={`${styles.chevron} ${expanded ? styles.expanded : ''}`}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}
