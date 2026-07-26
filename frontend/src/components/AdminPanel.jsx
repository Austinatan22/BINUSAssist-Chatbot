import { useEffect, useMemo, useRef, useState } from 'react'
import { Tabs } from '@base-ui/react/tabs'
import { ArrowDown, ArrowUp, Download, ExternalLink, Plus, RefreshCw, Trash2, TriangleAlert, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import ConfirmDialog from '@/components/ConfirmDialog'
import Header from '@/components/Header'
import { API_URL, apiFetch, authHeaders, jsonAuthHeaders } from '@/lib/api'

const ACCEPTED_EXTENSIONS = '.pdf,.docx,.xlsx,.csv'
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024

export default function AdminPanel({
  authHeader,
  isDark,
  setIsDark,
  username,
  openAdmin,
  handleSignOut,
  navigate,
  avatarVersion,
}) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [reindexing, setReindexing] = useState(false)
  const [url, setUrl] = useState('')
  const [addingUrl, setAddingUrl] = useState(false)
  const [starterQuestionsText, setStarterQuestionsText] = useState('')
  const [savingStarterQuestions, setSavingStarterQuestions] = useState(false)
  const [starterQuestionsSaved, setStarterQuestionsSaved] = useState(false)
  const [contacts, setContacts] = useState([])
  const [savingContacts, setSavingContacts] = useState(false)
  const [contactsSaved, setContactsSaved] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [reindexConfirmOpen, setReindexConfirmOpen] = useState(false)
  const [uploadConflict, setUploadConflict] = useState(null)
  const [toast, setToast] = useState('')
  const [sortKey, setSortKey] = useState('filename')
  const [sortDir, setSortDir] = useState('asc')
  const fileInputRef = useRef(null)
  const toastTimeoutRef = useRef(null)

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((dir) => (dir === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sortedDocuments = useMemo(() => {
    const sorted = [...documents].sort((a, b) => {
      const av = a[sortKey] ?? ''
      const bv = b[sortKey] ?? ''
      const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv
      return sortDir === 'asc' ? cmp : -cmp
    })
    return sorted
  }, [documents, sortKey, sortDir])

  const showToast = (message) => {
    clearTimeout(toastTimeoutRef.current)
    setToast(message)
    toastTimeoutRef.current = setTimeout(() => setToast(''), 3000)
  }

  const fetchStarterQuestions = async () => {
    try {
      const res = await fetch(`${API_URL}/config/starter-questions`)
      if (!res.ok) return
      setStarterQuestionsText((await res.json()).join('\n'))
    } catch {
      // non-critical — leave the textarea empty rather than blocking the admin panel
    }
  }

  const fetchFallbackContacts = async (header) => {
    try {
      const res = await fetch(`${API_URL}/admin/fallback-contacts`, { headers: authHeaders(header) })
      if (!res.ok) return
      setContacts(await res.json())
    } catch {
      // non-critical — leave the list empty rather than blocking the admin panel
    }
  }

  const updateContact = (index, field, value) => {
    setContacts((prev) => prev.map((c, i) => (i === index ? { ...c, [field]: value } : c)))
    setContactsSaved(false)
  }

  const addContact = () => {
    setContacts((prev) => [...prev, { role: '', name: '', email: '', whatsapp: '' }])
    setContactsSaved(false)
  }

  const removeContact = (index) => {
    setContacts((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== index)))
    setContactsSaved(false)
  }

  const handleSaveContacts = async (e) => {
    e.preventDefault()
    setSavingContacts(true)
    setContactsSaved(false)
    setError('')
    try {
      const res = await apiFetch(
        `${API_URL}/admin/fallback-contacts`,
        {
          method: 'PUT',
          headers: jsonAuthHeaders(authHeader),
          body: JSON.stringify({ contacts }),
        },
        'Save failed'
      )
      setContacts((await res.json()).contacts)
      setContactsSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSavingContacts(false)
    }
  }

  const fetchDocuments = async (header) => {
    setLoading(true)
    setError('')
    try {
      const res = await apiFetch(`${API_URL}/admin/documents`, { headers: authHeaders(header) }, 'Request failed')
      setDocuments(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Deliberate one-time async data fetch on mount, not a synchronous cascading setState
  // -- each fetch* function only calls setState after its own await. authHeader is set
  // once at login and doesn't change during this panel's lifetime; only run this on mount.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchDocuments(authHeader)
    fetchStarterQuestions()
    fetchFallbackContacts(authHeader)
    return () => clearTimeout(toastTimeoutRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleSaveStarterQuestions = async (e) => {
    e.preventDefault()
    const questions = starterQuestionsText.split('\n').map((q) => q.trim()).filter(Boolean)
    setSavingStarterQuestions(true)
    setStarterQuestionsSaved(false)
    setError('')
    try {
      await apiFetch(
        `${API_URL}/admin/starter-questions`,
        {
          method: 'PUT',
          headers: jsonAuthHeaders(authHeader),
          body: JSON.stringify({ questions }),
        },
        'Save failed'
      )
      setStarterQuestionsSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSavingStarterQuestions(false)
    }
  }

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (file.size > MAX_UPLOAD_BYTES) {
      setError('File exceeds 20MB limit.')
      return
    }
    await doUpload(file, false)
  }

  // Split out from handleUpload so the "replace it?" confirm dialog (triggered by a 409
  // conflict -- IMPROVEMENTS.md #5.3, e.g. uploading a newer catalog year alongside the
  // old one) can retry the same file with supersede=true without re-reading the input.
  const doUpload = async (file, supersede) => {
    setLoading(true)
    setError('')
    try {
      const body = new FormData()
      body.append('file', file)
      const query = supersede ? '?supersede=true' : ''
      const res = await fetch(`${API_URL}/admin/documents${query}`, {
        method: 'POST',
        headers: authHeaders(authHeader),
        body,
      })
      if (res.status === 409) {
        const data = await res.json().catch(() => ({}))
        setUploadConflict({ file, message: data.detail?.message ?? 'This file conflicts with an existing document.' })
        setLoading(false)
        return
      }
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Upload failed: ${res.status}`)
      }
      const result = await res.json()
      await fetchDocuments(authHeader)
      showToast(
        result.superseded_filename
          ? `"${file.name}" uploaded, replacing "${result.superseded_filename}".`
          : `"${file.name}" uploaded.`
      )
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const confirmUploadSupersede = () => {
    const { file } = uploadConflict
    setUploadConflict(null)
    doUpload(file, true)
  }

  const handleDelete = (filename) => setDeleteTarget(filename)

  const confirmDelete = async () => {
    const filename = deleteTarget
    setDeleteTarget(null)
    setLoading(true)
    setError('')
    try {
      await apiFetch(
        `${API_URL}/admin/documents?filename=${encodeURIComponent(filename)}`,
        { method: 'DELETE', headers: authHeaders(authHeader) },
        'Delete failed'
      )
      await fetchDocuments(authHeader)
      showToast(`"${filename}" deleted.`)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const handleAddUrl = async (e) => {
    e.preventDefault()
    if (!url.trim()) return
    setAddingUrl(true)
    setError('')
    try {
      await apiFetch(
        `${API_URL}/admin/documents/url`,
        {
          method: 'POST',
          headers: jsonAuthHeaders(authHeader),
          body: JSON.stringify({ url: url.trim() }),
        },
        'Add URL failed'
      )
      setUrl('')
      await fetchDocuments(authHeader)
    } catch (err) {
      setError(err.message)
    } finally {
      setAddingUrl(false)
    }
  }

  const handleReindex = () => setReindexConfirmOpen(true)

  const confirmReindex = async () => {
    setReindexConfirmOpen(false)
    setReindexing(true)
    setError('')
    try {
      const res = await apiFetch(
        `${API_URL}/admin/reindex`,
        { method: 'POST', headers: authHeaders(authHeader) },
        'Reindex failed'
      )
      const result = await res.json()
      await fetchDocuments(authHeader)
      if (result?.urls_failed?.length) {
        showToast(`Re-indexed, but ${result.urls_failed.length} URL(s) could not be re-fetched and have no cached content: ${result.urls_failed.join(', ')}`)
      } else if (result?.urls_from_cache?.length) {
        // Re-fetch failed but last-known-good cache kept the content -- softer signal than a
        // genuine loss, but worth surfacing so the admin can investigate/re-add.
        showToast(`Re-indexed. ${result.urls_from_cache.length} URL(s) failed to re-fetch and were served from cache (stale): ${result.urls_from_cache.join(', ')}`)
      } else {
        showToast('Knowledge base re-indexed.')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setReindexing(false)
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
        subtitle="Admin Panel"
      />
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto p-4 space-y-4">
          {error && <p className="text-sm text-destructive">{error}</p>}

          <Tabs.Root defaultValue="documents">
            <Tabs.List className="relative flex gap-1 border-b border-border mb-4">
              <Tabs.Tab
                value="documents"
                className="px-3 py-2 text-sm font-medium text-muted-foreground outline-none transition data-[active]:text-foreground"
              >
                Documents
              </Tabs.Tab>
              <Tabs.Tab
                value="starter-questions"
                className="px-3 py-2 text-sm font-medium text-muted-foreground outline-none transition data-[active]:text-foreground"
              >
                Starter Questions
              </Tabs.Tab>
              <Tabs.Tab
                value="fallback-contacts"
                className="px-3 py-2 text-sm font-medium text-muted-foreground outline-none transition data-[active]:text-foreground"
              >
                Fallback Contacts
              </Tabs.Tab>
              <Tabs.Indicator className="absolute bottom-0 h-[2px] w-[var(--active-tab-width)] translate-x-[var(--active-tab-left)] bg-accent transition-all duration-200" />
            </Tabs.List>

            <Tabs.Panel value="documents" className="space-y-4">
              <div className="border border-border rounded-md bg-card overflow-hidden">
                <div className="p-4 border-b border-border">
                  <h2 className="text-sm font-medium">Add Documents</h2>
                  <p className="text-xs text-muted-foreground">Upload files or scrape a web page into the knowledge base</p>
                </div>

                <div className="p-4 space-y-3">
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1" onClick={() => fileInputRef.current?.click()}>
                      <Upload size={14} className="mr-2" /> Upload document
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept={ACCEPTED_EXTENSIONS}
                      className="hidden"
                      onChange={handleUpload}
                    />
                    <Button variant="accent" onClick={handleReindex} disabled={reindexing}>
                      <RefreshCw size={14} className="mr-2" /> {reindexing ? 'Re-indexing…' : 'Re-index all'}
                    </Button>
                  </div>

                  <form onSubmit={handleAddUrl} className="flex gap-2">
                    <Input
                      type="url"
                      placeholder="https://binus.ac.id/..."
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                    />
                    <Button type="submit" variant="outline" disabled={addingUrl || !url.trim()}>
                      {addingUrl ? 'Adding…' : 'Add URL'}
                    </Button>
                  </form>
                </div>
              </div>

              <div className="border border-border rounded-md bg-card overflow-hidden">
                <div className="p-4 border-b border-border">
                  <h2 className="text-sm font-medium">Indexed Documents</h2>
                  <p className="text-xs text-muted-foreground">Documents currently in the knowledge base</p>
                </div>

                <table className="w-full text-sm table-fixed">
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border bg-muted/50">
                      <th className="py-2 px-3 w-auto">
                        <button onClick={() => handleSort('filename')} className="flex items-center gap-1 hover:text-foreground transition">
                          Filename
                          {sortKey === 'filename' && (sortDir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                        </button>
                      </th>
                      <th className="py-2 px-3 w-20">
                        <button onClick={() => handleSort('chunk_count')} className="flex items-center gap-1 hover:text-foreground transition">
                          Chunks
                          {sortKey === 'chunk_count' && (sortDir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                        </button>
                      </th>
                      <th className="py-2 px-3 w-32">
                        <button onClick={() => handleSort('ingested_at')} className="flex items-center gap-1 hover:text-foreground transition">
                          Ingested
                          {sortKey === 'ingested_at' && (sortDir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                        </button>
                      </th>
                      <th className="py-2 px-3 w-16"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedDocuments.map((doc) => (
                      <tr key={doc.filename} className="border-b border-border last:border-0 hover:bg-muted/40 transition">
                        <td className="py-2 px-3">
                          <div className="truncate" title={doc.filename}>{doc.filename}</div>
                        </td>
                        <td className="py-2 px-3">{doc.chunk_count}</td>
                        <td className="py-2 px-3 text-muted-foreground">
                          <span className="inline-flex items-center gap-1">
                            {doc.ingested_at?.slice(0, 10) ?? 'unknown'}
                            {doc.stale && (
                              <TriangleAlert
                                size={12}
                                className="text-amber-500"
                                role="img"
                                aria-label="Ingested over a year ago -- check for a newer source"
                              >
                                <title>Ingested over a year ago -- check for a newer source</title>
                              </TriangleAlert>
                            )}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-right">
                          <div className="flex items-center justify-end gap-3">
                            {(() => {
                              const isUrlSource = /^https?:\/\//.test(doc.filename)
                              return (
                                <a
                                  href={
                                    isUrlSource
                                      ? doc.filename
                                      : `${API_URL}/documents/${encodeURIComponent(doc.filename)}`
                                  }
                                  download={isUrlSource ? undefined : true}
                                  target={isUrlSource ? '_blank' : undefined}
                                  rel={isUrlSource ? 'noopener noreferrer' : undefined}
                                  className="text-muted-foreground hover:text-foreground transition"
                                  aria-label={isUrlSource ? `Open ${doc.filename}` : `Download ${doc.filename}`}
                                >
                                  {isUrlSource ? <ExternalLink size={14} /> : <Download size={14} />}
                                </a>
                              )
                            })()}
                            <button
                              onClick={() => handleDelete(doc.filename)}
                              className="text-muted-foreground hover:text-destructive"
                              aria-label={`Delete ${doc.filename}`}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {loading && <p className="p-4 text-xs text-muted-foreground">Loading…</p>}
                {!loading && documents.length === 0 && (
                  <p className="p-4 text-xs text-muted-foreground">No documents indexed yet.</p>
                )}
              </div>
            </Tabs.Panel>

            <Tabs.Panel value="starter-questions">
              <form onSubmit={handleSaveStarterQuestions} className="space-y-2 border border-border rounded-md p-4 bg-card">
                <h2 className="text-sm font-medium">Starter questions</h2>
                <p className="text-xs text-muted-foreground">One question per line. Shown as tappable chips on first load.</p>
                <Textarea
                  value={starterQuestionsText}
                  onChange={(e) => {
                    setStarterQuestionsText(e.target.value)
                    setStarterQuestionsSaved(false)
                  }}
                  rows={5}
                  className="text-sm"
                />
                <div className="flex items-center gap-2">
                  <Button type="submit" size="sm" disabled={savingStarterQuestions}>
                    {savingStarterQuestions ? 'Saving…' : 'Save'}
                  </Button>
                  {starterQuestionsSaved && <span className="text-xs text-green-600">Saved.</span>}
                </div>
              </form>
            </Tabs.Panel>

            <Tabs.Panel value="fallback-contacts">
              <form onSubmit={handleSaveContacts} className="space-y-3 border border-border rounded-md p-4 bg-card">
                <div>
                  <h2 className="text-sm font-medium">Fallback contacts</h2>
                  <p className="text-xs text-muted-foreground">
                    Shown to users when the bot can't answer or hits an error. At least one contact is required.
                  </p>
                </div>

                <div className="space-y-3">
                  {contacts.map((contact, i) => (
                    <div key={i} className="grid grid-cols-2 gap-2 border border-border rounded-md p-3 relative">
                      <button
                        type="button"
                        onClick={() => removeContact(i)}
                        disabled={contacts.length <= 1}
                        aria-label="Remove contact"
                        title={contacts.length <= 1 ? 'At least one contact is required' : undefined}
                        className="group absolute -top-3 -right-3 flex items-center justify-center h-7 w-7 rounded-full border border-border bg-card text-destructive disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none"
                      >
                        <span className="absolute inset-0 rounded-full bg-destructive/0 group-hover:bg-destructive/15 transition" />
                        <Trash2 size={14} className="relative" />
                      </button>
                      <Input
                        placeholder="Role (e.g. General Inquiries)"
                        value={contact.role}
                        onChange={(e) => updateContact(i, 'role', e.target.value)}
                        required
                        className="text-sm col-span-2"
                      />
                      <Input
                        placeholder="Name (e.g. Halo BINUS)"
                        value={contact.name}
                        onChange={(e) => updateContact(i, 'name', e.target.value)}
                        required
                        className="text-sm col-span-2"
                      />
                      <Input
                        placeholder="Email"
                        type="email"
                        value={contact.email}
                        onChange={(e) => updateContact(i, 'email', e.target.value)}
                        required
                        className="text-sm"
                      />
                      <Input
                        placeholder="WhatsApp link (optional)"
                        value={contact.whatsapp}
                        onChange={(e) => updateContact(i, 'whatsapp', e.target.value)}
                        className="text-sm"
                      />
                    </div>
                  ))}
                </div>

                <Button type="button" variant="outline" size="sm" onClick={addContact}>
                  <Plus size={14} className="mr-2" /> Add contact
                </Button>

                <div className="flex items-center gap-2">
                  <Button type="submit" size="sm" disabled={savingContacts}>
                    {savingContacts ? 'Saving…' : 'Save'}
                  </Button>
                  {contactsSaved && <span className="text-xs text-green-600">Saved.</span>}
                </div>
              </form>
            </Tabs.Panel>
          </Tabs.Root>
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-4 right-4 z-50 rounded-md border border-border bg-card px-4 py-2 text-sm text-foreground shadow-lg">
          {toast}
        </div>
      )}

      <ConfirmDialog
        open={deleteTarget != null}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete document"
        description={`Delete "${deleteTarget}" from the knowledge base? This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        onConfirm={confirmDelete}
      />
      <ConfirmDialog
        open={reindexConfirmOpen}
        onOpenChange={setReindexConfirmOpen}
        title="Re-index knowledge base"
        description='Re-index rebuilds the entire knowledge base from backend/documents/ and re-fetches every previously added URL. This can take a while. Continue?'
        confirmLabel="Re-index"
        onConfirm={confirmReindex}
      />
      <ConfirmDialog
        open={uploadConflict != null}
        onOpenChange={(open) => !open && setUploadConflict(null)}
        title="Replace existing document?"
        description={uploadConflict?.message ?? ''}
        confirmLabel="Replace"
        destructive
        onConfirm={confirmUploadSupersede}
      />
    </div>
  )
}
