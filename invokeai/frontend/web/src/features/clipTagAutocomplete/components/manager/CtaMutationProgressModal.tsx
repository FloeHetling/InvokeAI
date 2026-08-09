import {
  Modal,
  ModalBody,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Progress,
  Text,
  VStack,
} from '@invoke-ai/ui-library';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  isOpen: boolean;
  title: string;
  indeterminate?: boolean;
};

export const CtaMutationProgressModal = ({ isOpen, title, indeterminate = true }: Props) => {
  const { t } = useTranslation();
  const [show, setShow] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    clearTimeout(timerRef.current);
    if (!isOpen) {
      timerRef.current = setTimeout(() => setShow(false), 0);
      return;
    }
    timerRef.current = setTimeout(() => setShow(true), 400);
    return () => clearTimeout(timerRef.current);
  }, [isOpen]);

  if (!show) {
    return null;
  }

  return (
    <Modal isOpen={show} onClose={() => {}} closeOnOverlayClick={false} closeOnEsc={false} isCentered>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{title}</ModalHeader>
        <ModalBody>
          <VStack spacing={4}>
            <Text>{t('cta.pleaseWait')}</Text>
            {indeterminate ? <Progress size="xs" isIndeterminate w="100%" /> : <Progress size="xs" w="100%" />}
          </VStack>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
