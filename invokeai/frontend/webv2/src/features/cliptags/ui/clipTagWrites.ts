import { invalidateClipTagData } from '@features/cliptags/data/queries';
import { toaster } from '@platform/ui/toaster';
import { useMutation, useQueryClient } from '@tanstack/react-query';

/** Every write to the tag database carries this key, so the page can tell that one is in flight. */
export const CLIP_TAG_WRITE_KEY = ['cliptags', 'write'] as const;

/**
 * A write to the tag database. Whatever its outcome, the manager's reads are refreshed (a bulk operation can fail
 * after changing some rows), and the mutation stays pending until they have been, so a dialog never closes over
 * stale rows.
 */
export const useClipTagWrite = <TVariables, TResult>(mutationFn: (variables: TVariables) => Promise<TResult>) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn,
    mutationKey: CLIP_TAG_WRITE_KEY,
    onSettled: () => invalidateClipTagData(queryClient),
  });
};

export const notify = (type: 'error' | 'info' | 'success', title: string, description?: string): void => {
  toaster.create({ description, title, type });
};
