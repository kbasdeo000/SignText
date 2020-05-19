import React, {Component} from 'react';
import { BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import { Home } from './Home';
import { About } from './About';
import { Translate } from './Translate';
import { NoMatch } from './NoMatch';
import { Layout } from './components/Layout'; 
import { NavigationBar } from './components/NavigationBar'; 
import FadeIn from 'react-fade-in';

class App extends Component {
  render () {
    return (
      <React.Fragment> 
        <NavigationBar />
        <FadeIn />
        <Layout>
          <Router>
            <Switch>
              <Route exact path="/" component={Home} />
              <Route path="/about" component={About} />
              <Route path="/translate" component={Translate} />
              <Route component={NoMatch} />
            </Switch>
          </Router>
        </Layout>
      </React.Fragment>
    ); 
  }
}

export default App;
