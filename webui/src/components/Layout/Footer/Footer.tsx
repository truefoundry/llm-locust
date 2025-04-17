import { Container } from '@mui/material';

export default function Footer() {
  return (
    <Container
      maxWidth='xl'
      sx={{
        display: 'flex',
        height: 'var(--footer-height)',
        alignItems: 'center',
        justifyContent: 'flex-end',
      }}
    >
    </Container>
  );
}
