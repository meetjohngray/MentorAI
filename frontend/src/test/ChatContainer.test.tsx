import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatContainer } from '../components/ChatContainer';
import * as api from '../services/api';

// Mock the API module
vi.mock('../services/api', () => ({
  sendChatMessage: vi.fn(),
}));

describe('ChatContainer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders welcome message initially', () => {
    render(<ChatContainer />);

    expect(screen.getByText('Welcome')).toBeInTheDocument();
    expect(
      screen.getByText(/help you reflect on your experiences/i)
    ).toBeInTheDocument();
  });

  it('displays user message after sending', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage).mockResolvedValue({
      response: 'Test response',
      sources: [],
    });

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Hello mentor{enter}');

    expect(screen.getByText('Hello mentor')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();
  });

  it('displays assistant response after API call', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage).mockResolvedValue({
      response: 'I notice you mentioned patterns...',
      sources: [],
    });

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'What patterns do you see?{enter}');

    await waitFor(() => {
      expect(
        screen.getByText('I notice you mentioned patterns...')
      ).toBeInTheDocument();
    });
  });

  it('shows loading state while waiting for response', async () => {
    const user = userEvent.setup();
    // Create a promise that we can control
    let resolvePromise: (value: unknown) => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    vi.mocked(api.sendChatMessage).mockReturnValue(
      pendingPromise as Promise<{ response: string; sources: [] }>
    );

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Test message{enter}');

    // Should show loading indicator
    expect(screen.getByText('Reflecting...')).toBeInTheDocument();

    // Resolve the promise
    resolvePromise!({ response: 'Done', sources: [] });

    await waitFor(() => {
      expect(screen.queryByText('Reflecting...')).not.toBeInTheDocument();
    });
  });

  it('disables input while loading', async () => {
    const user = userEvent.setup();
    let resolvePromise: (value: unknown) => void;
    const pendingPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    vi.mocked(api.sendChatMessage).mockReturnValue(
      pendingPromise as Promise<{ response: string; sources: [] }>
    );

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Test{enter}');

    expect(input).toBeDisabled();

    resolvePromise!({ response: 'Done', sources: [] });

    await waitFor(() => {
      expect(input).not.toBeDisabled();
    });
  });

  it('displays error message on API failure', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage).mockRejectedValue(
      new Error('Network error')
    );

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Test message{enter}');

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('removes user message on error so they can retry', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage).mockRejectedValue(
      new Error('Network error')
    );

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'My message{enter}');

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    // User message should be removed
    expect(screen.queryByText('My message')).not.toBeInTheDocument();
  });

  it('passes conversation history to API', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage)
      .mockResolvedValueOnce({ response: 'First response', sources: [] })
      .mockResolvedValueOnce({ response: 'Second response', sources: [] });

    render(<ChatContainer />);

    const input = screen.getByRole('textbox');

    // First message
    await user.type(input, 'First question{enter}');
    await waitFor(() => {
      expect(screen.getByText('First response')).toBeInTheDocument();
    });

    // Second message
    await user.type(input, 'Follow up{enter}');
    await waitFor(() => {
      expect(screen.getByText('Second response')).toBeInTheDocument();
    });

    // Check that history was passed to second call
    expect(api.sendChatMessage).toHaveBeenLastCalledWith({
      message: 'Follow up',
      conversationHistory: [
        { role: 'user', content: 'First question' },
        { role: 'assistant', content: 'First response' },
      ],
    });
  });

  it('hides welcome message after first message', async () => {
    const user = userEvent.setup();
    vi.mocked(api.sendChatMessage).mockResolvedValue({
      response: 'Response',
      sources: [],
    });

    render(<ChatContainer />);

    expect(screen.getByText('Welcome')).toBeInTheDocument();

    const input = screen.getByRole('textbox');
    await user.type(input, 'Hello{enter}');

    expect(screen.queryByText('Welcome')).not.toBeInTheDocument();
  });
});
