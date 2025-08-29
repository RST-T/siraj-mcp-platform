'use client'

import { useUser, withPageAuthRequired } from '@auth0/nextjs-auth0/client'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  CreditCardIcon,
  KeyIcon,
  ChartBarIcon,
  BookOpenIcon,
  UserGroupIcon,
  PlusIcon
} from '@heroicons/react/24/outline'
import { apiClient } from '@/lib/api-client'
import { UsageChart } from '@/components/dashboard/usage-chart'
import { APIKeyManager } from '@/components/dashboard/api-key-manager'
import { CreditsPurchase } from '@/components/dashboard/credits-purchase'

function DashboardPage() {
  const { user, isLoading: userLoading } = useUser()

  const { data: credits, isLoading: creditsLoading } = useQuery({
    queryKey: ['user-credits'],
    queryFn: () => apiClient.get('/api/v1/credits').then(res => res.data),
    enabled: !!user
  })

  const { data: usage, isLoading: usageLoading } = useQuery({
    queryKey: ['user-usage'],
    queryFn: () => apiClient.get('/api/v1/usage').then(res => res.data),
    enabled: !!user
  })

  const { data: apiKeys, isLoading: keysLoading } = useQuery({
    queryKey: ['api-keys'],
    queryFn: () => apiClient.get('/api/v1/api-keys').then(res => res.data),
    enabled: !!user
  })

  if (userLoading || creditsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  const creditPercentage = credits ? (credits.balance / 5.0) * 100 : 0
  const monthlyUsagePercentage = credits ? (credits.monthly_free_used / 5.0) * 100 : 0

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg p-6 text-white"
      >
        <h1 className="text-2xl font-bold mb-2">
          Welcome back, {user?.name || user?.email}
        </h1>
        <p className="opacity-90">
          Continue your computational hermeneutics research with SIRAJ v6.1
        </p>
      </motion.div>

      {/* Quick Stats */}
      <div className="grid md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Credit Balance</CardTitle>
              <CreditCardIcon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${credits?.balance?.toFixed(2) || '0.00'}
              </div>
              <Progress value={creditPercentage} className="mt-2" />
              <p className="text-xs text-muted-foreground mt-1">
                {creditPercentage.toFixed(0)}% of monthly allowance
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">API Calls Today</CardTitle>
              <ChartBarIcon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {usage?.filter(u => {
                  const today = new Date().toDateString()
                  const usageDate = new Date(u.created_at).toDateString()
                  return today === usageDate
                }).length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Last 24 hours
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active API Keys</CardTitle>
              <KeyIcon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{apiKeys?.length || 0}</div>
              <p className="text-xs text-muted-foreground">
                Managing access
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Account Tier</CardTitle>
              <Badge variant="outline">Free</Badge>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Researcher</div>
              <p className="text-xs text-muted-foreground">
                Academic access
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <div className="grid lg:grid-cols-2 gap-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BookOpenIcon className="h-5 w-5 mr-2" />
                Quick Analysis
              </CardTitle>
              <CardDescription>
                Start a new computational hermeneutics analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <Button variant="outline" size="sm" asChild>
                  <Link href="/tools?tool=computational_hermeneutics_methodology">
                    Root Analysis
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/tools?tool=adaptive_semantic_architecture">
                    Semantic Mapping
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/tools?tool=community_sovereignty_protocols">
                    Cultural Validation
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/tools?tool=multi_paradigm_validation">
                    Paradigm Check
                  </Link>
                </Button>
              </div>
              <Button className="w-full" asChild>
                <Link href="/tools">
                  <PlusIcon className="h-4 w-4 mr-2" />
                  New Analysis
                </Link>
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <UserGroupIcon className="h-5 w-5 mr-2" />
                Community Activity
              </CardTitle>
              <CardDescription>
                Recent discussions and validations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                • New discussion on Arabic root k-t-b variations
              </div>
              <div className="text-sm text-muted-foreground">
                • Community validation for Hebrew cognates
              </div>
              <div className="text-sm text-muted-foreground">
                • Expert review pending for Quranic interpretation
              </div>
              <Button variant="outline" className="w-full" asChild>
                <Link href="/community">
                  View Community
                </Link>
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Detailed Sections */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Usage Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Usage Analytics</CardTitle>
              <CardDescription>Your API usage over the last 30 days</CardDescription>
            </CardHeader>
            <CardContent>
              {!usageLoading && usage ? (
                <UsageChart data={usage} />
              ) : (
                <div className="flex items-center justify-center h-64">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* API Key Management */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>API Key Management</CardTitle>
              <CardDescription>Manage your API keys for programmatic access</CardDescription>
            </CardHeader>
            <CardContent>
              {!keysLoading ? (
                <APIKeyManager keys={apiKeys || []} />
              ) : (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Credits Purchase */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.9 }}
      >
        <CreditsPurchase currentBalance={credits?.balance || 0} />
      </motion.div>
    </div>
  )
}

export default withPageAuthRequired(DashboardPage)