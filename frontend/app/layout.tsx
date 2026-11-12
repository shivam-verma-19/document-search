"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ErrorBoundary } from "../components/ErrorBoundary";
import AuthControls from "../components/AuthControls";
import { AuthProvider } from "../components/AuthProvider";
import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  const navItems = [
    { name: "Dashboard", href: "/" },
    { name: "Interactive Chat", href: "/chat" },
    { name: "Document Ingestion", href: "/upload" },
    { name: "Metrics & RAGAS", href: "/metrics" },
  ];

  return (
    <html lang="en">
      <head>
        <title>Hybrid RAG Platform</title>
        <meta name="description" content="Enterprise Hybrid RAG Search Platform" />
      </head>
      <body>
        <AuthProvider>
          <nav className="navbar">
            <Link href="/" className="nav-brand">
              <span>⚡</span>
              <span>Hybrid RAG Engine</span>
            </Link>
            <div className="nav-links">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`nav-link ${pathname === item.href ? "active" : ""}`}
                >
                  {item.name}
                </Link>
              ))}
              <AuthControls />
            </div>
          </nav>
          <main className="main-container">
            <ErrorBoundary>{children}</ErrorBoundary>
          </main>
        </AuthProvider>
      </body>
    </html>
  );
}
