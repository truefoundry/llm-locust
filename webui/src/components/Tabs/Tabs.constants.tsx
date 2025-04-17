import LogViewer from 'components/LogViewer/LogViewer';
import Reports from 'components/Reports/Reports';
import SwarmCharts from 'components/SwarmCharts/SwarmCharts';
import { LOG_VIEWER_KEY } from 'constants/logs';
import { ITab } from 'types/tab.types';

export const tabConfig = {
  charts: {
    component: SwarmCharts,
    key: 'charts',
    title: 'Charts',
  },
  reports: {
    component: Reports,
    key: 'reports',
    title: 'Download Data',
  },
  logs: {
    component: LogViewer,
    key: LOG_VIEWER_KEY,
    title: 'Logs',
  },
};

export const baseTabs: ITab[] = [
  tabConfig.charts,
  tabConfig.reports,
  tabConfig.logs,
];
