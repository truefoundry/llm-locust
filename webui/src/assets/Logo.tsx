export default function Logo() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <img
        alt="TrueFoundry Logo"
        height="35"
        src="https://app.truefoundry.com/assets/logo-CnOf81Q7.svg"
        width="35"
      />
      <span style={{ fontSize: '18px', fontWeight: '500' }}>LLM Benchmarking</span>
    </div>
  );
}
