import { Button, Flex, Spinner, Text, VStack } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

type Props = {
  isOpen: boolean;
  message: string;
  onRetry?: () => void;
  zIndex?: number;
};

export const CtaManagerOverlay = ({ isOpen, message, onRetry, zIndex = 10 }: Props) => {
  const { t } = useTranslation();

  if (!isOpen) {
    return null;
  }

  return (
    <Flex
      position="absolute"
      top={0}
      right={0}
      bottom={0}
      left={0}
      zIndex={zIndex}
      alignItems="center"
      justifyContent="center"
      borderRadius="md"
      style={{
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
      }}
    >
      <VStack
        role={onRetry ? 'alert' : 'status'}
        spacing={3}
        px={5}
        py={4}
        bg="base.800"
        borderWidth="1px"
        borderColor="base.600"
        borderRadius="lg"
        shadow="dark-lg"
      >
        {!onRetry && <Spinner size="md" thickness="2px" emptyColor="base.600" color="base.200" />}
        <Text fontSize="sm" textAlign="center">
          {message}
        </Text>
        {onRetry && (
          <Button size="sm" onClick={onRetry}>
            {t('cta.retry')}
          </Button>
        )}
      </VStack>
    </Flex>
  );
};
