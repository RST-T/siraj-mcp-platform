import './globals.css'
import { Inter } from 'next/font/google'
import { UserProvider } from '@auth0/nextjs-auth0/client'
import { QueryProvider } from '@/components/providers/query-provider'
import { Toaster } from '@/components/ui/toaster'
import { Navbar } from '@/components/layout/navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'SIRAJ v6.1 - Computational Hermeneutics Platform',
  description: 'Revolutionary computational hermeneutics platform integrating Islamic Tafsir, comparative linguistics, and modern NLP with cultural sovereignty protection.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <UserProvider>
        <body className={inter.className}>
          <QueryProvider>
            <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
              <Navbar />
              <main className="container mx-auto px-4 py-8">
                {children}
              </main>
              <Toaster />
            </div>
          </QueryProvider>
        </body>
      </UserProvider>
    </html>
  )
}