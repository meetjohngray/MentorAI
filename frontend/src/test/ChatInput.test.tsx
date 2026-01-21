import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatInput } from '../components/ChatInput';

describe('ChatInput', () => {
  it('renders input and button', () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('calls onSubmit with message when form is submitted', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Hello mentor');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(onSubmit).toHaveBeenCalledWith('Hello mentor');
  });

  it('calls onSubmit when Enter is pressed', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Hello mentor{enter}');

    expect(onSubmit).toHaveBeenCalledWith('Hello mentor');
  });

  it('does not submit on Shift+Enter (allows newlines)', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Line 1{shift>}{enter}{/shift}Line 2');

    expect(onSubmit).not.toHaveBeenCalled();
    expect(input).toHaveValue('Line 1\nLine 2');
  });

  it('clears input after submission', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    await user.type(input, 'Hello');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(input).toHaveValue('');
  });

  it('does not submit empty or whitespace-only messages', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    const button = screen.getByRole('button', { name: /send/i });

    // Empty message
    await user.click(button);
    expect(onSubmit).not.toHaveBeenCalled();

    // Whitespace only
    await user.type(input, '   ');
    await user.click(button);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('disables input and button when disabled prop is true', () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} disabled />);

    expect(screen.getByRole('textbox')).toBeDisabled();
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();
  });

  it('shows custom placeholder', () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} placeholder="Ask me anything..." />);

    expect(
      screen.getByPlaceholderText('Ask me anything...')
    ).toBeInTheDocument();
  });

  it('trims whitespace from message before submitting', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);

    const input = screen.getByRole('textbox');
    await user.type(input, '  Hello  ');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(onSubmit).toHaveBeenCalledWith('Hello');
  });
});
