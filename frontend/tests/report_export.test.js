// frontend/tests/report_export.test.js
// Unit tests for Stage B Report Agent and Export functionality in the frontend.
import { test, describe } from 'node:test'
import assert from 'node:assert/strict'
import { downloadRunExport, fetchRunReport } from '../src/services/api.js'

describe('Stage B Frontend Export & Report Contracts', () => {
  test('downloadRunExport rejects when runId is missing or empty', async () => {
    await assert.rejects(
      async () => {
        await downloadRunExport(null, 'markdown')
      },
      {
        message: 'A valid runId is required to export.',
      }
    )

    await assert.rejects(
      async () => {
        await downloadRunExport('', 'json')
      },
      {
        message: 'A valid runId is required to export.',
      }
    )
  })

  test('fetchRunReport throws expected error when fetch fails', async () => {
    // Mock global fetch to return 404 error
    const origFetch = globalThis.fetch
    globalThis.fetch = async () => ({
      ok: false,
      json: async () => ({ error: "Experiment run 'run_fake_123' not found." }),
    })

    try {
      await assert.rejects(
        async () => {
          await fetchRunReport('run_fake_123')
        },
        /Experiment run 'run_fake_123' not found/
      )
    } finally {
      globalThis.fetch = origFetch
    }
  })

  test('downloadRunExport triggers DOM download link with correct filename', async () => {
    let createdElement = null
    let clickCalled = false
    let removedChild = false

    const mockBlob = { size: 1024, type: 'text/markdown' }
    const origFetch = globalThis.fetch
    const origDocument = globalThis.document
    const origWindow = globalThis.window

    globalThis.fetch = async (url) => {
      assert(url.includes('/runs/run_valid_123/export/markdown'))
      return {
        ok: true,
        headers: {
          get: (header) => {
            if (header.toLowerCase() === 'content-disposition') {
              return 'attachment; filename="report_student_run_valid_123.md"'
            }
            return null
          },
        },
        blob: async () => mockBlob,
      }
    }

    globalThis.window = {
      URL: {
        createObjectURL: () => 'blob:http://localhost:5173/mock-blob',
        revokeObjectURL: () => {},
      },
    }

    globalThis.document = {
      createElement: (tag) => {
        createdElement = {
          tagName: tag,
          click: () => {
            clickCalled = true
          },
        }
        return createdElement
      },
      body: {
        appendChild: () => {},
        removeChild: () => {
          removedChild = true
        },
      },
    }

    try {
      const res = await downloadRunExport('run_valid_123', 'markdown')
      assert.equal(res.success, true)
      assert.equal(res.filename, 'report_student_run_valid_123.md')
      assert.equal(createdElement.download, 'report_student_run_valid_123.md')
      assert.equal(clickCalled, true)
      assert.equal(removedChild, true)
    } finally {
      globalThis.fetch = origFetch
      globalThis.document = origDocument
      globalThis.window = origWindow
    }
  })

  test('Phase 7 Markdown export shortcut contract propagates current run ID correctly', async () => {
    let capturedUrl = null
    const origFetch = globalThis.fetch
    const origDocument = globalThis.document
    const origWindow = globalThis.window

    globalThis.fetch = async (url) => {
      capturedUrl = url
      return {
        ok: true,
        headers: {
          get: (h) => (h.toLowerCase() === 'content-disposition' ? 'attachment; filename="report_eval_run_999.md"' : null),
        },
        blob: async () => ({ size: 500, type: 'text/markdown' }),
      }
    }

    globalThis.window = {
      URL: {
        createObjectURL: () => 'blob:http://localhost:5173/eval-blob',
        revokeObjectURL: () => {},
      },
    }

    globalThis.document = {
      createElement: () => ({ click: () => {} }),
      body: { appendChild: () => {}, removeChild: () => {} },
    }

    try {
      // Simulate Phase 7 action calling export with the active run ID
      const activeRunId = 'run_20261023_active_evaluation_442'
      const res = await downloadRunExport(activeRunId, 'markdown')
      assert.equal(res.success, true)
      assert.equal(res.filename, 'report_eval_run_999.md')
      assert.ok(capturedUrl.includes(`/runs/${encodeURIComponent(activeRunId)}/export/markdown`))

      // Simulate invocation with null/missing activeRunId (must reject)
      await assert.rejects(async () => {
        await downloadRunExport(null, 'markdown')
      }, /A valid runId is required to export/)
    } finally {
      globalThis.fetch = origFetch
      globalThis.document = origDocument
      globalThis.window = origWindow
    }
  })
})
