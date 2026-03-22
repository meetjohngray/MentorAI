import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConversationSidebar } from '../components/ConversationSidebar';
import * as api from '../services/api';

vi.mock('../services/api', () => ({
  getConversations: vi.fn(),
  deleteConversation: vi.fn(),
}));

const defaultProps = {
  activeConversationId: null,
  onSelectConversation: vi.fn(),
  onNewConversation: vi.fn(),
  refreshTrigger: 0,
};

describe('ConversationSidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders empty state when no conversations', async () => {
    vi.mocked(api.getConversations).mockResolvedValue([]);

    render(<ConversationSidebar {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText(/no conversations yet/i)).toBeInTheDocument();
    });
  });

  it('renders conversation list', async () => {
    vi.mocked(api.getConversations).mockResolvedValue([
      {
        id: 'conv-1',
        title: 'First conversation',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        message_count: 4,
        preview: 'Hello mentor',
      },
      {
        id: 'conv-2',
        title: 'Second conversation',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        message_count: 2,
      },
    ]);

    render(<ConversationSidebar {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('First conversation')).toBeInTheDocument();
      expect(screen.getByText('Second conversation')).toBeInTheDocument();
    });
  });

  it('calls onSelectConversation when conversation clicked', async () => {
    const user = userEvent.setup();
    const onSelectConversation = vi.fn();
    vi.mocked(api.getConversations).mockResolvedValue([
      {
        id: 'conv-1',
        title: 'Test conversation',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        message_count: 2,
      },
    ]);

    render(
      <ConversationSidebar
        {...defaultProps}
        onSelectConversation={onSelectConversation}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Test conversation')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Test conversation'));

    expect(onSelectConversation).toHaveBeenCalledWith('conv-1');
  });

  it('calls onNewConversation when new button clicked', async () => {
    const user = userEvent.setup();
    const onNewConversation = vi.fn();
    vi.mocked(api.getConversations).mockResolvedValue([]);

    render(
      <ConversationSidebar
        {...defaultProps}
        onNewConversation={onNewConversation}
      />
    );

    await user.click(screen.getByText('+ New'));

    expect(onNewConversation).toHaveBeenCalled();
  });

  it('renders header and new button', () => {
    vi.mocked(api.getConversations).mockResolvedValue([]);

    render(<ConversationSidebar {...defaultProps} />);

    expect(screen.getByText('Conversations')).toBeInTheDocument();
    expect(screen.getByText('+ New')).toBeInTheDocument();
  });
});
