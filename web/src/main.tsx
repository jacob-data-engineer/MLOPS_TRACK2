import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App.tsx";
import { AuthProvider } from "./context/AuthContext.tsx";
import { NavGuardProvider } from "./context/NavGuardContext.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <NavGuardProvider>
          <App />
        </NavGuardProvider>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
