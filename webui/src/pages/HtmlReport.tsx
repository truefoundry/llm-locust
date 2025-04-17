import { Box, Typography, Container } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';

import ResponseTimeTable from 'components/ResponseTimeTable/ResponseTimeTable';
import { SwarmCharts } from 'components/SwarmCharts/SwarmCharts';
import { INITIAL_THEME } from 'constants/theme';
import theme from 'redux/slice/theme.slice';
import createTheme from 'styles/theme';
import { IReport } from 'types/swarm.types';
import { formatLocaleString } from 'utils/date';

const muiTheme = createTheme(window.theme || INITIAL_THEME);
const isDarkMode = (window.theme || INITIAL_THEME) === 'dark';

const reportStore = configureStore({
  reducer: combineReducers({ theme }),
  preloadedState: { theme: { isDarkMode } },
});

export default function HtmlReport({
  locustfile,
  startTime,
  endTime,
  charts,
  host,
  responseTimeStatistics,
}: IReport) {
  return (
    <Provider store={reportStore}>
      <ThemeProvider theme={muiTheme}>
        <CssBaseline />
        <Container maxWidth='xl'>
          <Box sx={{ display: 'flex', flexDirection: 'column', rowGap: 4, py: 4 }}>
            <Box>
              <Typography component='h1' mb={1} noWrap variant='h3'>
                {locustfile}
              </Typography>
              <Typography component='h2' mb={1} noWrap variant='h5'>
                {host}
              </Typography>
              <Typography component='h2' mb={1} noWrap variant='h5'>
                {`${formatLocaleString(startTime)} - ${formatLocaleString(endTime)}`}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', flexDirection: 'column', rowGap: 4 }}>
              {!!responseTimeStatistics.length && (
                <Box>
                  <Typography component='h2' mb={1} noWrap variant='h4'>
                    Response Time Statistics
                  </Typography>
                  <ResponseTimeTable responseTimes={responseTimeStatistics} />
                </Box>
              )}

              <Box>
                <Typography component='h2' mb={1} noWrap variant='h4'>
                  Charts
                </Typography>
                <SwarmCharts charts={charts} />
              </Box>
            </Box>
          </Box>
        </Container>
      </ThemeProvider>
    </Provider>
  );
}
