import { Link, List, ListItem } from '@mui/material';
import { connect } from 'react-redux';

import { IRootState } from 'redux/store';

function Reports() {
  return (
    <List sx={{ display: 'flex', flexDirection: 'column' }}>
      <ListItem>
        <Link href='/stats/requests/csv'>Download requests CSV</Link>
      </ListItem>
    </List>
  );
}

export default connect(({ swarm }: IRootState) => ({
  extendedCsvFiles: swarm.extendedCsvFiles,
  statsHistoryEnabled: swarm.statsHistoryEnabled,
}))(Reports);
