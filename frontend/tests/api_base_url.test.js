import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { resolveApiBaseUrl } from '../src/services/api.js'

describe('Deployment Readiness: API Base URL Resolution', () => {
  it('defaults to /api when environment variable is unset or undefined', () => {
    assert.strictEqual(resolveApiBaseUrl(undefined), '/api')
  })

  it('defaults to /api when environment variable is empty or whitespace', () => {
    assert.strictEqual(resolveApiBaseUrl(''), '/api')
    assert.strictEqual(resolveApiBaseUrl('   '), '/api')
  })

  it('appends /api if root deployment domain is provided without trailing slash', () => {
    assert.strictEqual(
      resolveApiBaseUrl('https://datainvest-api.onrender.com'),
      'https://datainvest-api.onrender.com/api'
    )
  })

  it('appends /api if root deployment domain is provided with trailing slash', () => {
    assert.strictEqual(
      resolveApiBaseUrl('https://datainvest-api.onrender.com/'),
      'https://datainvest-api.onrender.com/api'
    )
  })

  it('avoids duplicate /api when /api is explicitly provided', () => {
    assert.strictEqual(
      resolveApiBaseUrl('https://datainvest-api.onrender.com/api'),
      'https://datainvest-api.onrender.com/api'
    )
  })

  it('avoids duplicate /api and strips trailing slash when /api/ is provided', () => {
    assert.strictEqual(
      resolveApiBaseUrl('https://datainvest-api.onrender.com/api/'),
      'https://datainvest-api.onrender.com/api'
    )
  })

  it('preserves relative /api path', () => {
    assert.strictEqual(resolveApiBaseUrl('/api'), '/api')
    assert.strictEqual(resolveApiBaseUrl('/api/'), '/api')
  })
})

