import { Alert } from '@chakra-ui/react';

/** An inline failure notice for a dialog that stays open so the action can be retried. */
export const ErrorAlert = ({ message }: { message: string }) => (
  <Alert.Root status="error" variant="surface">
    <Alert.Indicator />
    <Alert.Content>
      <Alert.Description overflowWrap="anywhere">{message}</Alert.Description>
    </Alert.Content>
  </Alert.Root>
);
