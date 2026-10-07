import type { ReactNode } from 'react';

import { Center, Spinner } from '@chakra-ui/react';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** The loading and failure states a manager tab shows in place of its content. */
export const QueryState = ({
  children,
  errorTitle,
  query,
}: {
  /** Rendered once the data has arrived. */
  children: ReactNode;
  errorTitle: string;
  query: { data: unknown; error: unknown; isError: boolean; isPending: boolean; refetch: () => unknown };
}) => {
  const { t } = useTranslation();
  const { refetch } = query;
  const handleRetry = useCallback(() => void refetch(), [refetch]);

  if (query.isPending) {
    return (
      <Center aria-label={t('common.loading')} flex="1" minH="32" role="status">
        <Spinner color="fg.subtle" size="lg" />
      </Center>
    );
  }

  // A failed refresh keeps the rows already shown.
  if (query.isError && query.data === undefined) {
    return (
      <EmptyState danger description={getApiErrorMessage(query.error, '')} title={errorTitle}>
        <Button variant="outline" onClick={handleRetry}>
          {t('common.retry')}
        </Button>
      </EmptyState>
    );
  }

  return children;
};
