import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import AdminLogin from "@/components/AdminLogin";
import AdminPanel from "@/components/AdminPanel";
import ChatPanel from "@/components/ChatPanel";
import Header from "@/components/Header";
import Profile from "@/components/Profile";
import SourcePanel from "@/components/SourcePanel";
import { useChat } from "@/hooks/useChat";

const SOURCES_ROW_GAP = 15;

function viewForPath(pathname) {
  if (pathname === "/admin") return "admin";
  if (pathname === "/profile") return "profile";
  if (pathname === "/login") return "login";
  return "chat";
}

const AUTH_STORAGE_KEY = "adminAuth";

function loadStoredAuth() {
  try {
    return JSON.parse(sessionStorage.getItem(AUTH_STORAGE_KEY)) ?? {};
  } catch {
    return {};
  }
}

export default function App() {
  const { messages, sendMessage, regenerate, clearConversation, isStreaming, stopStreaming } = useChat();
  const [view, setView] = useState(() => viewForPath(window.location.pathname));
  // Persisted to sessionStorage (not localStorage) so a refresh/new tab in the same
  // browser session keeps the session, but it doesn't outlive closing the tab — these
  // are HTTP Basic credentials (base64, not a revocable token), so we don't want them
  // lingering indefinitely on disk.
  const [authHeader, setAuthHeader] = useState(() => loadStoredAuth().authHeader ?? null);
  const [username, setUsername] = useState(() => loadStoredAuth().username ?? null);
  const [role, setRole] = useState(() => loadStoredAuth().role ?? null);
  const [avatarVersion, setAvatarVersion] = useState(0);

  useEffect(() => {
    const onPopState = () => setView(viewForPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  // Visiting /admin or /profile directly (typed URL, refresh, bookmark) without a live
  // auth header (none stored, or a fresh browser session) bounces to the login page
  // instead of rendering a panel that has no credentials to fetch with.
  // The redirect target depends on the URL side effect below running first
  // (replaceState), so it can't be computed during render; the extra re-render this
  // causes is a one-off on auth-loss, not a steady-state cost.
  useEffect(() => {
    if ((view === "admin" || view === "profile") && !authHeader) {
      window.history.replaceState({}, "", "/login");
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setView("login");
    }
  }, [view, authHeader]);

  const navigate = (path) => {
    window.history.pushState({}, "", path);
    setView(viewForPath(path));
  };

  const openAdmin = () => navigate(authHeader ? "/admin" : "/login");
  const goToChat = () => navigate("/");

  const persistAuth = (header, authUsername, authRole) => {
    sessionStorage.setItem(
      AUTH_STORAGE_KEY,
      JSON.stringify({ authHeader: header, username: authUsername, role: authRole })
    );
  };

  const handleLoginSuccess = (header, loggedInUsername, loggedInRole) => {
    setAuthHeader(header);
    setUsername(loggedInUsername);
    setRole(loggedInRole);
    persistAuth(header, loggedInUsername, loggedInRole);
    navigate("/admin");
  };

  const handleSignOut = () => {
    setAuthHeader(null);
    setUsername(null);
    setRole(null);
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
    navigate("/");
  };

  const handleAvatarChanged = () => setAvatarVersion((v) => v + 1);
  const handleCredentialsUpdated = (header, newUsername) => {
    setAuthHeader(header);
    setUsername(newUsername);
    persistAuth(header, newUsername, role);
  };

  const [activeMessageIndex, setActiveMessageIndex] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const [isDark, setIsDark] = useState(() => {
    const stored = localStorage.getItem("theme");
    if (stored) return stored === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  const [sourcesWidth, setSourcesWidth] = useState(null);
  const [suppressWidthTransition, setSuppressWidthTransition] = useState(true);
  const rowRef = useRef(null);
  const resetTransitionFrameRef = useRef(null);

  const collapseSources = () => setSourcesCollapsed(true);
  const expandSources = () => setSourcesCollapsed(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
    localStorage.setItem("theme", isDark ? "dark" : "light");
  }, [isDark]);

  // Measure the row's content width and derive the sources panel's expanded
  // width in px, so it can stay a constant size (never reflowing its text)
  // while only the visible aside box animates between collapsed/expanded.
  // Resize-driven width updates skip the transition (suppressWidthTransition)
  // so zooming/resizing the viewport snaps instantly instead of animating —
  // only explicit collapse/expand toggles should ever animate the width.
  //
  // The reset back to false is debounced against the observer itself (a ref-tracked
  // double rAF, re-armed on every measure() call) rather than a separate effect keyed
  // on state — ResizeObserver reliably fires more than once around mount/layout
  // settling, and racing an independent effect against that caused the reset to
  // sometimes get clobbered by a second firing, permanently stuck at "no transition".
  useLayoutEffect(() => {
    const row = rowRef.current;
    if (!row) return;
    const measure = () => {
      const contentWidth =
        row.clientWidth -
        parseFloat(getComputedStyle(row).paddingLeft) -
        parseFloat(getComputedStyle(row).paddingRight);
      setSuppressWidthTransition(true);
      setSourcesWidth((contentWidth - SOURCES_ROW_GAP) * 0.2);

      if (resetTransitionFrameRef.current) {
        cancelAnimationFrame(resetTransitionFrameRef.current);
      }
      resetTransitionFrameRef.current = requestAnimationFrame(() => {
        resetTransitionFrameRef.current = requestAnimationFrame(() => {
          setSuppressWidthTransition(false);
        });
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(row);
    return () => {
      observer.disconnect();
      if (resetTransitionFrameRef.current) {
        cancelAnimationFrame(resetTransitionFrameRef.current);
      }
    };
  }, [view]);

  const latestSourcedIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].sources?.length) return i;
    }
    return null;
  }, [messages]);

  const effectiveIndex = activeMessageIndex ?? latestSourcedIndex;
  const activeSources =
    effectiveIndex != null ? (messages[effectiveIndex]?.sources ?? []) : [];

  const handleCiteClick = (messageIndex, sourceId) => {
    setActiveMessageIndex(messageIndex);
    // sourceId is null for the generic "X sources — view" button (as opposed to a
    // specific citation number) -- leave whatever card is currently open alone in that
    // case, rather than forcing every card closed.
    if (sourceId != null) {
      setHighlightId(sourceId);
    }
    setSheetOpen(true);
    expandSources();
  };

  const handleToggleSource = (sourceId) => {
    setHighlightId((prev) => (prev === sourceId ? null : sourceId));
  };

  if (view === "login") {
    return <AdminLogin onSuccess={handleLoginSuccess} onClose={goToChat} />;
  }

  if (view === "admin") {
    return (
      <AdminPanel
        authHeader={authHeader}
        onClose={goToChat}
        isDark={isDark}
        setIsDark={setIsDark}
        username={username}
        openAdmin={openAdmin}
        handleSignOut={handleSignOut}
        navigate={navigate}
        avatarVersion={avatarVersion}
      />
    );
  }

  if (view === "profile") {
    return (
      <Profile
        authHeader={authHeader}
        username={username}
        role={role}
        isDark={isDark}
        setIsDark={setIsDark}
        openAdmin={openAdmin}
        handleSignOut={handleSignOut}
        navigate={navigate}
        avatarVersion={avatarVersion}
        onAvatarChanged={handleAvatarChanged}
        onCredentialsUpdated={handleCredentialsUpdated}
      />
    );
  }

  return (
    <div className="flex flex-col h-screen">
      <Header
        isDark={isDark}
        setIsDark={setIsDark}
        authHeader={authHeader}
        username={username}
        openAdmin={openAdmin}
        handleSignOut={handleSignOut}
        navigate={navigate}
        avatarVersion={avatarVersion}
      />
      <div className="flex-1 overflow-hidden flex bg-background">
        <div ref={rowRef} className="flex w-full gap-[15px] p-4">
          <main
            className="flex-1 min-w-0 rounded-xl border border-border bg-linear-[135deg] from-primary/5 via-card to-accent/5 shadow-sm overflow-hidden p-4"
            style={{ contain: "paint" }}
          >
            <ChatPanel
              messages={messages}
              sendMessage={sendMessage}
              regenerate={regenerate}
              isStreaming={isStreaming}
              onStop={stopStreaming}
              onCiteClick={handleCiteClick}
              onNewChat={clearConversation}
            />
          </main>
          <aside
            className="hidden md:block shrink-0 rounded-xl border border-border bg-linear-[135deg] from-primary/5 via-card to-accent/5 shadow-sm overflow-hidden"
            style={{
              width: sourcesCollapsed ? 58 : sourcesWidth ?? "20%",
              transition: suppressWidthTransition ? "none" : "width 300ms ease-in-out",
              contain: "paint",
            }}
          >
            <div className="relative w-full h-full">
              <div
                className={`absolute inset-0 transition-opacity duration-300 ease-in-out ${
                  sourcesCollapsed ? "opacity-100" : "opacity-0 pointer-events-none"
                }`}
              >
                <button
                  onClick={expandSources}
                  aria-label="Show sources"
                  className="absolute top-4 right-4 text-muted-foreground hover:text-foreground hover:bg-muted/40 rounded-md transition"
                >
                  <span
                    className="material-symbols-outlined"
                    style={{ fontSize: 20 }}
                  >
                    right_panel_open
                  </span>
                </button>
                <div className="absolute inset-x-3 top-12 border-b border-border" />
                <div className="absolute inset-x-0 top-16 flex flex-col items-center gap-2 overflow-y-auto px-2 pb-2">
                  {activeSources.map((source) => (
                    <button
                      key={source.id}
                      onClick={() => {
                        setHighlightId(source.id);
                        expandSources();
                      }}
                      aria-label={`Show source ${source.id}`}
                      className="flex items-center justify-center h-7 min-w-7 px-1.5 shrink-0 rounded-sm bg-accent/15 text-accent text-xs font-medium hover:bg-accent/25 transition"
                    >
                      {source.id}
                    </button>
                  ))}
                </div>
              </div>
              <div
                className={`absolute inset-y-0 right-0 p-4 transition-opacity duration-300 ease-in-out ${
                  sourcesCollapsed ? "opacity-0 pointer-events-none" : "opacity-100"
                }`}
                style={{ width: sourcesWidth ?? "100%" }}
              >
                <SourcePanel
                  sources={activeSources}
                  highlightId={highlightId}
                  onToggle={handleToggleSource}
                  onCollapse={collapseSources}
                />
              </div>
            </div>
          </aside>
        </div>
      </div>
      <div
        className={`md:hidden fixed inset-x-0 bottom-0 z-50 h-[60vh] bg-card/90 backdrop-blur-md border-t border-border rounded-t-xl shadow-2xl transition-transform duration-300 ease-in-out ${
          sheetOpen ? "translate-y-0" : "translate-y-full pointer-events-none"
        }`}
      >
        <SourcePanel
          sources={activeSources}
          highlightId={highlightId}
          onToggle={handleToggleSource}
          onClose={() => setSheetOpen(false)}
        />
      </div>
      <footer className="border-t px-4 py-2 text-xs text-center text-muted-foreground bg-muted">
        ⚠ AI-generated answers — verify critical information with faculty.
      </footer>
    </div>
  );
}
