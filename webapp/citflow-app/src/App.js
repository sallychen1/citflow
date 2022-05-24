import React, { Component } from 'react';
import './App.css';
import { Grid, Image , Icon,Segment,Divider } from 'semantic-ui-react'


export default class App extends Component {
  constructor(props)  {
    super(props)

    this.state = {
      // floor : 0,
      apiResponse: ""
    }
    // this.updateFloor = this.updateFloor.bind(this);
    // this.updateFloor();
    // this.timer = setInterval(this.updateFloor, 1000);
  }

  callAPI() {
    fetch("http://localhost:1337")
        .then(res => res.text())
        .then(res => this.setState({ apiResponse: res }));
  }

  componentWillMount() {
      this.callAPI();
  }

  // updateFloor(){
  //   // var newFloor = 
  //   this.setState({
  //     // floor: newfloor
  //   })
  // }
    render(){
      return (
        <div>
          <Segment> 
          <Grid columns={2} divided>
            <Grid.Row>
              <Grid.Column>
                <Image src='https://react.semantic-ui.com/images/wireframe/white-image.png' size='medium' bordered />
                <Icon name='caret square up' />
                On Floor: {this.state.apiResponse}
              </Grid.Column>
              <Grid.Column>
                <Image src='https://react.semantic-ui.com/images/wireframe/white-image.png' size='medium' bordered />
                <Icon name='caret square down' />
                On Floor: {this.state.apiResponse}
              </Grid.Column>
            </Grid.Row>
          </Grid>
          <Divider vertical></Divider>
          </Segment>
        </div>
      )
    }
}