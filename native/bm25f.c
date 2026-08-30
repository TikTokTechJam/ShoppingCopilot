/* Native SQLite FTS5 auxiliary function for the repository's BM25F formula.
 *
 * The function receives the immutable query configuration as trailing SQL
 * arguments:
 *   k1, then (field_weight, field_b, average_field_length) per field.
 *
 * FTS5 supplies the per-row term frequencies and field lengths. xQueryPhrase
 * supplies document frequency once per MATCH expression, cached in FTS5
 * auxiliary data for the duration of that query.
 */

#include <math.h>
#include <sqlite3ext.h>
#include <stdio.h>
#include <stdlib.h>

SQLITE_EXTENSION_INIT1

#define BM25F_FIELD_COUNT 6
#define BM25F_CONFIG_VALUES (1 + BM25F_FIELD_COUNT * 3)

typedef struct Bm25fQueryData {
  int nPhrase;
  double *aIdf;
} Bm25fQueryData;

typedef struct PhraseCountContext {
  int count;
} PhraseCountContext;

static int bm25fCountPhrase(
    const Fts5ExtensionApi *pApi,
    Fts5Context *pFts,
    void *pUserData) {
  PhraseCountContext *pCount = (PhraseCountContext *)pUserData;
  (void)pApi;
  (void)pFts;
  pCount->count += 1;
  return SQLITE_OK;
}

static void bm25fDeleteQueryData(void *pValue) {
  Bm25fQueryData *pData = (Bm25fQueryData *)pValue;
  if (pData != NULL) {
    sqlite3_free(pData->aIdf);
    sqlite3_free(pData);
  }
}

static int bm25fQueryData(
    const Fts5ExtensionApi *pApi,
    Fts5Context *pFts,
    Bm25fQueryData **ppData) {
  Bm25fQueryData *pData = (Bm25fQueryData *)pApi->xGetAuxdata(pFts, 0);
  if (pData != NULL) {
    *ppData = pData;
    return SQLITE_OK;
  }

  pData = (Bm25fQueryData *)sqlite3_malloc64(sizeof(*pData));
  if (pData == NULL) return SQLITE_NOMEM;
  pData->nPhrase = pApi->xPhraseCount(pFts);
  pData->aIdf = (double *)sqlite3_malloc64(
      sizeof(double) * (sqlite3_uint64)(pData->nPhrase > 0 ? pData->nPhrase : 1));
  if (pData->aIdf == NULL) {
    sqlite3_free(pData);
    return SQLITE_NOMEM;
  }

  sqlite3_int64 nDocument = 0;
  int rc = pApi->xRowCount(pFts, &nDocument);
  if (rc != SQLITE_OK) {
    bm25fDeleteQueryData(pData);
    return rc;
  }

  for (int iPhrase = 0; iPhrase < pData->nPhrase; ++iPhrase) {
    PhraseCountContext count = {0};
    rc = pApi->xQueryPhrase(pFts, iPhrase, &count, bm25fCountPhrase);
    if (rc != SQLITE_OK) {
      bm25fDeleteQueryData(pData);
      return rc;
    }
    /* This is log(1 + (N-df+0.5)/(df+0.5)), matching the Python reference. */
    double numerator = (double)nDocument - (double)count.count + 0.5;
    double denominator = (double)count.count + 0.5;
    pData->aIdf[iPhrase] = log1p(numerator / denominator);
  }

  rc = pApi->xSetAuxdata(pFts, pData, bm25fDeleteQueryData);
  if (rc != SQLITE_OK) {
    bm25fDeleteQueryData(pData);
    return rc;
  }
  *ppData = pData;
  return SQLITE_OK;
}

static int bm25fCalculate(
    const Fts5ExtensionApi *pApi,
    Fts5Context *pFts,
    int nVal,
    sqlite3_value **apVal,
    double **ppLevels,
    int *pnLevels,
    double *pTotal) {
  if (nVal != BM25F_CONFIG_VALUES) {
    return SQLITE_MISUSE;
  }
  if (pApi->xColumnCount(pFts) != BM25F_FIELD_COUNT) {
    return SQLITE_MISMATCH;
  }

  Bm25fQueryData *pQuery = NULL;
  int rc = bm25fQueryData(pApi, pFts, &pQuery);
  if (rc != SQLITE_OK) return rc;

  double k1 = sqlite3_value_double(apVal[0]);
  double aWeight[BM25F_FIELD_COUNT];
  double aB[BM25F_FIELD_COUNT];
  double aAverageLength[BM25F_FIELD_COUNT];
  for (int iField = 0; iField < BM25F_FIELD_COUNT; ++iField) {
    int offset = 1 + iField * 3;
    aWeight[iField] = sqlite3_value_double(apVal[offset]);
    aB[iField] = sqlite3_value_double(apVal[offset + 1]);
    aAverageLength[iField] = sqlite3_value_double(apVal[offset + 2]);
  }

  int nInstance = 0;
  rc = pApi->xInstCount(pFts, &nInstance);
  if (rc != SQLITE_OK) {
    return rc;
  }
  int nCells = pQuery->nPhrase * BM25F_FIELD_COUNT;
  int *aTf = (int *)sqlite3_malloc64(
      sizeof(int) * (sqlite3_uint64)(nCells > 0 ? nCells : 1));
  if (aTf == NULL) return SQLITE_NOMEM;
  for (int i = 0; i < nCells; ++i) aTf[i] = 0;

  for (int iInstance = 0; iInstance < nInstance; ++iInstance) {
    int iPhrase = 0;
    int iColumn = 0;
    int iOffset = 0;
    rc = pApi->xInst(pFts, iInstance, &iPhrase, &iColumn, &iOffset);
    if (rc != SQLITE_OK) {
      sqlite3_free(aTf);
      return rc;
    }
    (void)iOffset;
    if (iPhrase >= 0 && iPhrase < pQuery->nPhrase &&
        iColumn >= 0 && iColumn < BM25F_FIELD_COUNT) {
      aTf[iPhrase * BM25F_FIELD_COUNT + iColumn] += 1;
    }
  }

  int nLevels = 1;
  for (int iPhrase = 0; iPhrase < pQuery->nPhrase; ++iPhrase) {
    int phraseSize = pApi->xPhraseSize(pFts, iPhrase);
    if (phraseSize > nLevels) nLevels = phraseSize;
  }
  double *aLevels = (double *)sqlite3_malloc64(
      sizeof(double) * (sqlite3_uint64)nLevels);
  if (aLevels == NULL) {
    sqlite3_free(aTf);
    return SQLITE_NOMEM;
  }
  for (int i = 0; i < nLevels; ++i) aLevels[i] = 0.0;

  for (int iPhrase = 0; iPhrase < pQuery->nPhrase; ++iPhrase) {
    int phraseSize = pApi->xPhraseSize(pFts, iPhrase);
    if (phraseSize < 1) phraseSize = 1;
    if (phraseSize > nLevels) phraseSize = nLevels;
    double weightedTf = 0.0;
    for (int iField = 0; iField < BM25F_FIELD_COUNT; ++iField) {
      int tf = aTf[iPhrase * BM25F_FIELD_COUNT + iField];
      if (tf == 0) continue;
      int fieldLength = 0;
      rc = pApi->xColumnSize(pFts, iField, &fieldLength);
      if (rc != SQLITE_OK) {
        sqlite3_free(aTf);
        sqlite3_free(aLevels);
        return rc;
      }
      double normalizer = 1.0;
      if (aAverageLength[iField] > 0.0) {
        normalizer = 1.0 - aB[iField] + aB[iField] *
            ((double)fieldLength / aAverageLength[iField]);
      }
      weightedTf += aWeight[iField] * (double)tf / normalizer;
    }
    if (weightedTf != 0.0) {
      double contribution = pQuery->aIdf[iPhrase] *
          ((k1 + 1.0) * weightedTf) /
          (k1 + weightedTf);
      aLevels[phraseSize - 1] += contribution;
    }
  }
  sqlite3_free(aTf);
  double total = 0.0;
  for (int i = 0; i < nLevels; ++i) total += aLevels[i];
  *ppLevels = aLevels;
  *pnLevels = nLevels;
  *pTotal = total;
  return SQLITE_OK;
}

static void bm25fFunction(
    const Fts5ExtensionApi *pApi,
    Fts5Context *pFts,
    sqlite3_context *pCtx,
    int nVal,
    sqlite3_value **apVal) {
  double *aLevels = NULL;
  int nLevels = 0;
  double total = 0.0;
  int rc = bm25fCalculate(
      pApi, pFts, nVal, apVal, &aLevels, &nLevels, &total);
  (void)nLevels;
  if (rc != SQLITE_OK) {
    if (rc == SQLITE_MISUSE) {
      sqlite3_result_error(
          pCtx, "bm25f expects k1 plus 3 values per field", -1);
    } else if (rc == SQLITE_MISMATCH) {
      sqlite3_result_error(pCtx, "bm25f FTS column count mismatch", -1);
    } else {
      sqlite3_result_error_code(pCtx, rc);
    }
    return;
  }
  sqlite3_free(aLevels);
  sqlite3_result_double(pCtx, total);
}

static void bm25fLevelsFunction(
    const Fts5ExtensionApi *pApi,
    Fts5Context *pFts,
    sqlite3_context *pCtx,
    int nVal,
    sqlite3_value **apVal) {
  double *aLevels = NULL;
  int nLevels = 0;
  double total = 0.0;
  int rc = bm25fCalculate(
      pApi, pFts, nVal, apVal, &aLevels, &nLevels, &total);
  (void)total;
  if (rc != SQLITE_OK) {
    if (rc == SQLITE_MISUSE) {
      sqlite3_result_error(
          pCtx, "bm25f_levels expects k1 plus 3 values per field", -1);
    } else if (rc == SQLITE_MISMATCH) {
      sqlite3_result_error(pCtx, "bm25f_levels FTS column count mismatch", -1);
    } else {
      sqlite3_result_error_code(pCtx, rc);
    }
    return;
  }

  sqlite3_uint64 capacity = (sqlite3_uint64)nLevels * 40 + 32;
  char *result = (char *)sqlite3_malloc64(capacity);
  if (result == NULL) {
    sqlite3_free(aLevels);
    sqlite3_result_error_code(pCtx, SQLITE_NOMEM);
    return;
  }
  int offset = snprintf(result, (size_t)capacity, "{");
  for (int i = 0; i < nLevels; ++i) {
    offset += snprintf(
        result + offset, (size_t)capacity - (size_t)offset,
        "%s\"%d\":%.17g", i == 0 ? "" : ",", i + 1, aLevels[i]);
  }
  (void)snprintf(
      result + offset, (size_t)capacity - (size_t)offset, "}");
  sqlite3_free(aLevels);
  sqlite3_result_text(pCtx, result, -1, sqlite3_free);
}

static int bm25fApiFromDb(sqlite3 *db, fts5_api **ppApi) {
  sqlite3_stmt *pStmt = NULL;
  int rc = sqlite3_prepare_v2(db, "SELECT fts5(?1)", -1, &pStmt, NULL);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_bind_pointer(pStmt, 1, (void *)ppApi, "fts5_api_ptr", NULL);
  if (rc == SQLITE_OK) {
    (void)sqlite3_step(pStmt);
  }
  sqlite3_finalize(pStmt);
  return rc;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_bm25f_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi) {
  SQLITE_EXTENSION_INIT2(pApi);
  fts5_api *pFts5 = NULL;
  int rc = bm25fApiFromDb(db, &pFts5);
  if (rc != SQLITE_OK || pFts5 == NULL) {
    if (pzErrMsg != NULL) *pzErrMsg = sqlite3_mprintf("FTS5 API unavailable");
    return rc == SQLITE_OK ? SQLITE_ERROR : rc;
  }
  rc = pFts5->xCreateFunction(
      pFts5, "bm25f", NULL, bm25fFunction, NULL);
  if (rc == SQLITE_OK) {
    rc = pFts5->xCreateFunction(
        pFts5, "bm25f_levels", NULL, bm25fLevelsFunction, NULL);
  }
  return rc;
}

/* SQLite's default load_extension symbol derivation strips digits from the
 * basename, so bm25f.dylib is probed as sqlite3_bmf_init on some builds. */
int sqlite3_bmf_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi) {
  return sqlite3_bm25f_init(db, pzErrMsg, pApi);
}
