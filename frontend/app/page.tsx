'use client'

import { useUser } from '@auth0/nextjs-auth0/client'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  BookOpenIcon, 
  GlobeAltIcon, 
  AcademicCapIcon,
  SparklesIcon,
  ArrowRightIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'

export default function HomePage() {
  const { user, isLoading } = useUser()

  const features = [
    {
      title: "Computational Hermeneutics",
      description: "Revolutionary framework integrating Islamic Tafsir, comparative linguistics, and modern NLP",
      icon: BookOpenIcon,
      cost: "$0.05"
    },
    {
      title: "Adaptive Semantic Architecture",
      description: "Dynamic 5-tier semantic mapping with cultural adaptation and context awareness",
      icon: SparklesIcon,
      cost: "$0.10"
    },
    {
      title: "Cultural Sovereignty Protocols",
      description: "Ensures community authority over cultural knowledge with validation workflows",
      icon: GlobeAltIcon,
      cost: "$0.08"
    },
    {
      title: "Multi-Paradigm Validation",
      description: "Convergence validation across Traditional, Scientific, and Computational paradigms",
      icon: AcademicCapIcon,
      cost: "$0.12"
    }
  ]

  const benefits = [
    "Decode Semitic roots with academic precision",
    "Access peer-reviewed community knowledge",
    "Integrate with existing research workflows",
    "Export to academic writing tools",
    "Cultural sovereignty protection built-in"
  ]

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Revolutionary
            <span className="text-blue-600 block">Computational Hermeneutics</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Bridge traditional Islamic scholarship with cutting-edge computational linguistics. 
            Decode Semitic roots, validate cultural knowledge, and maintain academic rigor.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
            {!user ? (
              <>
                <Button size="lg" asChild>
                  <Link href="/api/auth/login">
                    Start Free Trial
                    <ArrowRightIcon className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="#features">Learn More</Link>
                </Button>
              </>
            ) : (
              <>
                <Button size="lg" asChild>
                  <Link href="/dashboard">
                    Go to Dashboard
                    <ArrowRightIcon className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="/community">Explore Community</Link>
                </Button>
              </>
            )}
          </div>

          <div className="flex justify-center space-x-8 text-sm text-gray-500">
            <div className="flex items-center">
              <CheckCircleIcon className="h-4 w-4 text-green-500 mr-1" />
              $5 Free Credits Monthly
            </div>
            <div className="flex items-center">
              <CheckCircleIcon className="h-4 w-4 text-green-500 mr-1" />
              Academic Citations Included
            </div>
            <div className="flex items-center">
              <CheckCircleIcon className="h-4 w-4 text-green-500 mr-1" />
              Cultural Sovereignty Protected
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Four Methodology Tools
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6 mb-12">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 * (index + 1) }}
              >
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between mb-2">
                      <feature.icon className="h-8 w-8 text-blue-600" />
                      <Badge variant="secondary">{feature.cost}</Badge>
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button variant="outline" size="sm" asChild>
                      <Link href={user ? "/tools" : "/api/auth/login"}>
                        Try Now
                      </Link>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Benefits Section */}
      <section className="py-12">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-6">
              Why Choose SIRAJ?
            </h2>
            <div className="space-y-4">
              {benefits.map((benefit, index) => (
                <motion.div
                  key={benefit}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.1 * (index + 1) }}
                  className="flex items-start"
                >
                  <CheckCircleIcon className="h-5 w-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">{benefit}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-none">
              <CardHeader>
                <CardTitle className="text-2xl text-center">Ready to Begin?</CardTitle>
                <CardDescription className="text-center">
                  Join researchers worldwide using SIRAJ for computational hermeneutics
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center space-y-4">
                <div className="text-3xl font-bold text-blue-600">$5</div>
                <div className="text-sm text-gray-600">Free credits every month</div>
                <div className="text-xs text-gray-500">
                  100 methodology queries • Community access • Academic citations
                </div>
                {!user ? (
                  <Button size="lg" className="w-full" asChild>
                    <Link href="/api/auth/login">
                      Create Free Account
                      <ArrowRightIcon className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                ) : (
                  <Button size="lg" className="w-full" asChild>
                    <Link href="/dashboard">
                      Access Dashboard
                      <ArrowRightIcon className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Community Preview */}
      <section className="py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            Join the Academic Community
          </h2>
          <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
            Connect with scholars, validate cultural knowledge, and contribute to the growing 
            database of computational hermeneutics research with full source attribution.
          </p>
          
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <Card>
              <CardHeader className="text-center">
                <div className="text-2xl font-bold text-blue-600">500+</div>
                <div className="text-sm text-gray-600">Verified Scholars</div>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader className="text-center">
                <div className="text-2xl font-bold text-green-600">2,000+</div>
                <div className="text-sm text-gray-600">Root Analyses</div>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader className="text-center">
                <div className="text-2xl font-bold text-purple-600">50+</div>
                <div className="text-sm text-gray-600">Universities</div>
              </CardHeader>
            </Card>
          </div>

          <Button variant="outline" size="lg" asChild>
            <Link href="/community">
              Explore Community
              <ArrowRightIcon className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </motion.div>
      </section>
    </div>
  )
}