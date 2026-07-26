import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { API_URL, authHeaders } from '@/lib/api'

export default function AdminLogin({ onSuccess, onClose }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState('')

  const handleLogin = async (e) => {
    e.preventDefault()
    const header = `Basic ${btoa(`${username}:${password}`)}`
    setAuthError('')
    try {
      const res = await fetch(`${API_URL}/admin/profile`, { headers: authHeaders(header) })
      if (res.status === 401) {
        setAuthError('Incorrect username or password.')
        return
      }
      if (res.status === 429) {
        // The backend already sends a readable detail message, but its wait time is
        // always phrased in raw seconds (e.g. "in 897 seconds") -- reformat using the
        // Retry-After header so a 15-minute lockout doesn't read like a typo.
        const retryAfter = Number(res.headers.get('Retry-After'))
        const wait =
          retryAfter > 60
            ? `${Math.ceil(retryAfter / 60)} minute${Math.ceil(retryAfter / 60) === 1 ? '' : 's'}`
            : `${retryAfter || 'a few'} second${retryAfter === 1 ? '' : 's'}`
        setAuthError(`Too many failed attempts. Try again in ${wait}.`)
        return
      }
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Something went wrong (error ${res.status}). Please try again.`)
      }
      const data = await res.json()
      onSuccess(header, data.username, data.role)
    } catch (err) {
      setAuthError(
        err instanceof TypeError
          ? 'Could not reach the server. Please make sure the backend is running.'
          : err.message
      )
    }
  }

  return (
    <div className="flex flex-col h-screen items-center justify-center p-4 bg-background">
      <div className="w-full max-w-md border border-border rounded-md bg-card overflow-hidden shadow-xl">
        <div className="p-4 flex flex-col items-start text-left gap-2">
          <img src="/binus-logo.png" alt="BINUS University" className="h-10 w-auto dark:hidden" />
          <img src="/binus-logo-dark.png" alt="BINUS University" className="hidden h-10 w-auto dark:block" />
          <h2 className="text-2xl font-semibold text-foreground">Log In as Admin</h2>
          <p className="text-xs text-muted-foreground">
            Sign in with your admin account to manage the chatbot
          </p>
        </div>

        <form onSubmit={handleLogin} className="p-4">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="admin-username" className="text-sm font-medium text-foreground">
                Username
              </label>
              <Input
                id="admin-username"
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                autoComplete="username"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label htmlFor="admin-password" className="text-sm font-medium text-foreground">
                Admin password
              </label>
              <Input
                id="admin-password"
                type="password"
                placeholder="Admin password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </div>
          </div>
          {authError && <p className="text-xs text-destructive mt-2">{authError}</p>}
          <div className="flex flex-col gap-2 mt-9">
            <Button type="submit" className="w-full h-10">Log In as Admin</Button>
            <Button type="button" variant="outline" onClick={onClose}>Back to chat</Button>
          </div>
        </form>
      </div>
    </div>
  )
}
