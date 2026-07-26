import { Menu } from "@base-ui/react/menu";
import { CircleUserRound, Moon, Sun } from "lucide-react";
import { API_URL } from "@/lib/api";

export default function Header({
  isDark,
  setIsDark,
  authHeader,
  username,
  openAdmin,
  handleSignOut,
  navigate,
  avatarVersion = 0,
  subtitle = "SoCS Chatbot",
}) {
  return (
    <div className="h-[55px] bg-card border-b border-border shadow-sm px-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/")}
          className="text-[22px] tracking-tight [font-family:'Inter_Variable']"
        >
          <span className="font-black text-foreground">BINUS</span>
          <span className="font-semibold text-accent">ASSIST</span>
        </button>
        <div className="h-5 w-px bg-border" />
        <span className="text-[14px] font-regular tracking-wide uppercase text-muted-foreground">
          {subtitle}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => setIsDark((prev) => !prev)}
          className="text-muted-foreground hover:text-primary transition"
          aria-label="Toggle dark mode"
        >
          {isDark ? <Sun size={24} /> : <Moon size={24} />}
        </button>
        {authHeader ? (
          <Menu.Root>
            <Menu.Trigger
              className="flex items-center gap-1.5 text-muted-foreground hover:text-primary transition"
              aria-label="Account"
            >
              <img
                key={avatarVersion}
                src={`${API_URL}/avatar/${encodeURIComponent(username)}?v=${avatarVersion}`}
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = "/default-avatar.jpg";
                }}
                alt=""
                className="h-6 w-6 rounded-full object-cover"
              />
              <span className="text-sm font-medium">{username}</span>
            </Menu.Trigger>
            <Menu.Portal>
              <Menu.Positioner align="end" sideOffset={32} className="z-50">
                <Menu.Popup className="min-w-[220px] rounded-lg border border-border bg-card/95 backdrop-blur-md shadow-lg p-1.5 text-base">
                  <Menu.Item
                    onClick={() => navigate("/profile")}
                    className="px-4 py-2.5 rounded-sm cursor-pointer outline-none hover:bg-muted/60 data-[highlighted]:bg-muted/60"
                  >
                    View Profile
                  </Menu.Item>
                  <Menu.Item
                    onClick={openAdmin}
                    className="px-4 py-2.5 rounded-sm cursor-pointer outline-none hover:bg-muted/60 data-[highlighted]:bg-muted/60"
                  >
                    Admin Panel
                  </Menu.Item>
                  <div className="my-1 h-px bg-border" />
                  <Menu.Item
                    onClick={handleSignOut}
                    className="px-4 py-2.5 rounded-sm cursor-pointer outline-none hover:bg-muted/60 data-[highlighted]:bg-muted/60"
                  >
                    Sign out
                  </Menu.Item>
                </Menu.Popup>
              </Menu.Positioner>
            </Menu.Portal>
          </Menu.Root>
        ) : (
          <button
            onClick={openAdmin}
            className="flex items-center gap-1.5 text-muted-foreground hover:text-primary transition"
            aria-label="Admin"
          >
            <CircleUserRound size={24} />
            <span className="text-sm font-medium">Login</span>
          </button>
        )}
      </div>
    </div>
  );
}
