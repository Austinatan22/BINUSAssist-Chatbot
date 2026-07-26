import { useCallback, useEffect, useRef, useState } from 'react'
import { API_URL } from '@/lib/api'

// Conversation persistence (IMPROVEMENTS.md #9.1) -- localStorage, not sessionStorage:
// unlike the admin auth header (App.jsx), this is just chat text, not a credential, so
// there's no reason it shouldn't survive closing the tab the way a real chat app's
// history would.
const MESSAGES_STORAGE_KEY = 'chatMessages'

function loadStoredMessages() {
  try {
    const parsed = JSON.parse(localStorage.getItem(MESSAGES_STORAGE_KEY))
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function useChat() {
  const [messages, setMessages] = useState(loadStoredMessages)
  const [isStreaming, setIsStreaming] = useState(false)
  // Holds the in-flight request's controller so stopStreaming can cancel it -- a ref,
  // not state, since it's only ever read/written from event handlers/callbacks, never
  // rendered.
  const abortControllerRef = useRef(null)

  useEffect(() => {
    localStorage.setItem(MESSAGES_STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  const updateLastMessage = (updater) => {
    setMessages((prev) => {
      const next = [...prev]
      next[next.length - 1] = updater(next[next.length - 1])
      return next
    })
  }

  // Shared by sendMessage and regenerate -- baseMessages is passed explicitly rather
  // than read from the messages state closure, since regenerate needs to send with the
  // LAST turn already stripped out (the answer being regenerated), and relying on a
  // setMessages call to land before this runs would race the state update.
  const runChat = useCallback(
    async (baseMessages, text) => {
      const trimmed = text.trim()
      if (!trimmed || isStreaming) return

      // History sent to the backend is everything said so far, not including the
      // question being asked right now. Only role/content travel over the wire; a
      // message with no content (e.g. a prior turn that errored out before any tokens
      // arrived) is dropped rather than sent as a blank turn.
      const history = baseMessages
        .filter((m) => m.content)
        .map((m) => ({ role: m.role, content: m.content }))

      setMessages([
        ...baseMessages,
        { role: 'user', content: trimmed },
        { role: 'assistant', content: '', sources: [], fallback: false, followUps: [], contacts: [] },
      ])
      setIsStreaming(true)

      const controller = new AbortController()
      abortControllerRef.current = controller

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: trimmed, history }),
          signal: controller.signal,
        })

        if (res.status === 429) {
          throw new Error('RATE_LIMITED')
        }
        if (!res.ok || !res.body) {
          throw new Error(`Request failed: ${res.status}`)
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const events = buffer.split('\n\n')
          buffer = events.pop()

          for (const event of events) {
            const line = event.trim()
            if (!line.startsWith('data:')) continue

            const data = JSON.parse(line.slice(5).trim())
            if (data.type === 'token') {
              updateLastMessage((msg) => ({ ...msg, content: msg.content + data.content }))
            } else if (data.type === 'done') {
              updateLastMessage((msg) => ({
                ...msg,
                sources: data.sources || [],
                fallback: !!data.fallback,
                followUps: data.follow_ups || [],
                // Only present on a fallback -- the escalation contacts, sent as
                // structured data so they render as a card rather than a wall of text
                // inside the message (see backend generation._fallback_events).
                contacts: data.contacts || [],
              }))
            }
          }
        }
      } catch (err) {
        // A user-initiated stop (stopStreaming) aborts the fetch, which rejects with
        // this specific error -- not a real failure, so it must NOT overwrite whatever
        // partial answer already streamed in with an error message.
        if (err.name !== 'AbortError') {
          const fallback =
            err.message === 'RATE_LIMITED'
              ? "You've sent too many messages. Please wait a bit before trying again."
              : 'Connection error. Please make sure the backend server is running.'
          updateLastMessage((msg) => ({ ...msg, content: msg.content || fallback }))
        }
      } finally {
        abortControllerRef.current = null
        setIsStreaming(false)
      }
    },
    [isStreaming]
  )

  const sendMessage = useCallback((text) => runChat(messages, text), [runChat, messages])

  // Cancels the in-flight request (Tier 1 UI: a stop button during streaming). A no-op
  // if nothing is streaming.
  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  // Regenerate/retry (IMPROVEMENTS.md #9.2): re-asks the last user question with the
  // same preceding history, replacing the last (possibly errored, possibly just
  // unhelpful) assistant answer rather than appending a duplicate turn.
  const regenerate = useCallback(() => {
    if (isStreaming || messages.length < 2) return
    const lastUser = messages[messages.length - 2]
    if (lastUser?.role !== 'user') return
    runChat(messages.slice(0, -2), lastUser.content)
  }, [isStreaming, messages, runChat])

  // Explicit reset for the persisted conversation (#9.1 made a refresh keep it, so
  // there needs to be an intentional way to start over instead).
  const clearConversation = useCallback(() => setMessages([]), [])

  return { messages, sendMessage, regenerate, clearConversation, isStreaming, stopStreaming }
}
