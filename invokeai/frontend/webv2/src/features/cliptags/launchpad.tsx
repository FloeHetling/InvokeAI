import { Center, Spinner } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

/** Launchpad entry kept separate so the editor's initial chunk does not load the manager. */
const LazyClipTagsPage = lazy(() => import('./ui/ClipTagsPage').then((module) => ({ default: module.ClipTagsPage })));

const FALLBACK = (
  <Center h="full">
    <Spinner color="fg.muted" size="lg" />
  </Center>
);

export const ClipTagsPage = () => (
  <Suspense fallback={FALLBACK}>
    <LazyClipTagsPage />
  </Suspense>
);
