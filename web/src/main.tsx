import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App.tsx";
import { RouteBoundary } from "./components/RouteBoundary.tsx";
import { AuthProvider } from "./context/AuthContext.tsx";
import { NavGuardProvider } from "./context/NavGuardContext.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* The last resort. `RouteBoundary` inside the Layout catches a page;
        this catches the shell around it -- the auth provider, the nav,
        the build badge -- so even then the screen says what broke instead
        of going white. */}
    <RouteBoundary>
      <BrowserRouter>
        <AuthProvider>
          <NavGuardProvider>
            <App />
          </NavGuardProvider>
        </AuthProvider>
      </BrowserRouter>
    </RouteBoundary>
  </StrictMode>,
);
