import { connect, useSelector } from 'react-redux';

import LineChart from 'components/LineChart/LineChart';
import { IRootState } from 'redux/store';
import { ICharts } from 'types/ui.types';


export function SwarmCharts({ charts }: { charts: ICharts }) {
  const renderableSwarmCharts = useSelector(({ ui }) => ui?.renderableSwarmCharts);
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '1rem'
    }}>
      {renderableSwarmCharts?.map((lineChartProps, index) => (
        <LineChart<ICharts> key={`swarm-chart-${lineChartProps?.title || index}`} {...lineChartProps} charts={charts} />
      ))}
    </div>
  );
}

const storeConnector = ({ ui: { charts } }: IRootState) => ({ charts });

export default connect(storeConnector)(SwarmCharts);
