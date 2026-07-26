import { useRef, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import Header from '@/components/Header'
import { API_URL, apiFetch, authHeaders, jsonAuthHeaders } from '@/lib/api'

const MAX_AVATAR_BYTES = 5 * 1024 * 1024
const ACCEPTED_AVATAR_TYPES = 'image/jpeg,image/png,image/webp'

function SettingsRow({ label, helper, children, last = false }) {
  return (
    <div className={`grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-2 sm:gap-4 p-4 ${last ? '' : 'border-b border-border'}`}>
      <label className="text-sm font-semibold text-foreground pt-1.5">{label}</label>
      <div className="space-y-1">
        {children}
        {helper && <p className="text-xs text-muted-foreground">{helper}</p>}
      </div>
    </div>
  )
}

export default function Profile({
  authHeader,
  username,
  role,
  isDark,
  setIsDark,
  openAdmin,
  handleSignOut,
  navigate,
  avatarVersion,
  onAvatarChanged,
  onCredentialsUpdated,
}) {
  const [usernameInput, setUsernameInput] = useState(username ?? '')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [avatarFile, setAvatarFile] = useState(null)
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [avatarRemoved, setAvatarRemoved] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  const usernameChanged = usernameInput.trim() !== (username ?? '')
  const isDirty = Boolean(avatarFile) || avatarRemoved || usernameChanged || newPassword.length > 0 || confirmPassword.length > 0

  const handleAvatarPick = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.')
      return
    }
    if (file.size > MAX_AVATAR_BYTES) {
      setError('Image exceeds 5MB limit.')
      return
    }
    setError('')
    setSaved(false)
    setAvatarRemoved(false)
    setAvatarFile(file)
    setAvatarPreview(URL.createObjectURL(file))
  }

  const handleRemoveAvatar = () => {
    setError('')
    setSaved(false)
    setAvatarFile(null)
    setAvatarPreview(null)
    setAvatarRemoved(true)
  }

  const handleSave = async (e) => {
    e.preventDefault()
    setError('')
    setSaved(false)

    const trimmedUsername = usernameInput.trim()
    if (!trimmedUsername) {
      setError('Username cannot be empty.')
      return
    }
    if (newPassword || confirmPassword) {
      if (newPassword.length < 8) {
        setError('New password must be at least 8 characters.')
        return
      }
      if (newPassword !== confirmPassword) {
        setError('New password and confirmation do not match.')
        return
      }
    }

    setSaving(true)
    try {
      // Username/password first, so a rename takes effect before any avatar change below
      // uploads/deletes under whichever username is current at that point.
      let activeHeader = authHeader

      if (usernameChanged || newPassword) {
        await apiFetch(
          `${API_URL}/admin/profile`,
          {
            method: 'PUT',
            headers: jsonAuthHeaders(authHeader),
            body: JSON.stringify({
              ...(usernameChanged ? { username: trimmedUsername } : {}),
              ...(newPassword ? { new_password: newPassword } : {}),
            }),
          },
          'Update failed'
        )

        // Basic Auth has no session of its own — the header itself encodes the
        // password, so to keep using it after a username-only change we recover
        // the still-current password from the existing header rather than
        // storing plaintext separately anywhere.
        const decoded = atob(authHeader.replace('Basic ', ''))
        const currentPassword = decoded.slice(decoded.indexOf(':') + 1)
        const finalPassword = newPassword || currentPassword
        activeHeader = `Basic ${btoa(`${trimmedUsername}:${finalPassword}`)}`
        onCredentialsUpdated(activeHeader, trimmedUsername)
        setNewPassword('')
        setConfirmPassword('')
      }

      if (avatarFile) {
        const body = new FormData()
        body.append('file', avatarFile)
        await apiFetch(
          `${API_URL}/admin/avatar`,
          { method: 'POST', headers: authHeaders(activeHeader), body },
          'Avatar upload failed'
        )
        setAvatarFile(null)
        setAvatarPreview(null)
        onAvatarChanged()
      } else if (avatarRemoved) {
        await apiFetch(
          `${API_URL}/admin/avatar`,
          { method: 'DELETE', headers: authHeaders(activeHeader) },
          'Removing avatar failed'
        )
        setAvatarRemoved(false)
        onAvatarChanged()
      }

      setSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <Header
        isDark={isDark}
        setIsDark={setIsDark}
        authHeader={authHeader}
        username={username}
        openAdmin={openAdmin}
        handleSignOut={handleSignOut}
        navigate={navigate}
        avatarVersion={avatarVersion}
        subtitle="Profile"
      />
      <form onSubmit={handleSave} className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto p-4 space-y-4">
          <div className="flex items-center gap-4 border border-border rounded-md p-4 bg-card">
            <img
              key={avatarPreview ?? (avatarRemoved ? 'removed' : avatarVersion)}
              src={
                avatarRemoved
                  ? '/default-avatar.jpg'
                  : avatarPreview ?? `${API_URL}/avatar/${encodeURIComponent(username)}?v=${avatarVersion}`
              }
              onError={(e) => {
                e.target.onerror = null
                e.target.src = '/default-avatar.jpg'
              }}
              alt=""
              className="h-20 w-20 rounded-full object-cover border border-border shrink-0"
            />
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Button type="button" variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()}>
                  Update Profile Picture
                </Button>
                <button
                  type="button"
                  onClick={handleRemoveAvatar}
                  aria-label="Remove profile picture"
                  className="text-muted-foreground hover:text-destructive transition"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_AVATAR_TYPES}
                className="hidden"
                onChange={handleAvatarPick}
              />
              <p className="text-xs text-muted-foreground">Must be JPEG, PNG, or WEBP and cannot exceed 5MB.</p>
            </div>
          </div>

          <div className="border border-border rounded-md bg-card overflow-hidden">
            <SettingsRow label="Role" helper="Roles are assigned by an administrator and can't be changed here" last>
              <p className="text-sm text-foreground">{role ? role.charAt(0).toUpperCase() + role.slice(1) : ''}</p>
            </SettingsRow>
          </div>

          <div className="border border-border rounded-md bg-card overflow-hidden">
            <div className="p-4 border-b border-border">
              <h2 className="text-sm font-medium">Profile Settings</h2>
              <p className="text-xs text-muted-foreground">Change identifying details for your account</p>
            </div>

            <SettingsRow label="Username" helper="You may update your username">
              <Input
                type="text"
                value={usernameInput}
                onChange={(e) => {
                  setUsernameInput(e.target.value)
                  setSaved(false)
                }}
                autoComplete="username"
              />
            </SettingsRow>

            <SettingsRow label="New Password" helper="Leave blank to keep your current password">
              <Input
                type="password"
                value={newPassword}
                onChange={(e) => {
                  setNewPassword(e.target.value)
                  setSaved(false)
                }}
                autoComplete="new-password"
              />
            </SettingsRow>

            <SettingsRow label="Confirm New Password" helper="Re-enter the new password to confirm">
              <Input
                type="password"
                value={confirmPassword}
                onChange={(e) => {
                  setConfirmPassword(e.target.value)
                  setSaved(false)
                }}
                autoComplete="new-password"
              />
            </SettingsRow>

            <div className="p-4 flex items-center justify-between gap-3">
              <div>
                {error && <p className="text-sm text-destructive">{error}</p>}
                {saved && !error && <p className="text-sm text-green-600">Changes saved.</p>}
              </div>
              <Button type="submit" disabled={!isDirty || saving}>
                {saving ? 'Saving…' : 'Save Changes'}
              </Button>
            </div>
          </div>

          <div className="flex justify-end">
            <Button type="button" variant="outline" onClick={handleSignOut}>
              Log out
            </Button>
          </div>
        </div>
      </form>
    </div>
  )
}
