import React from 'react';
import {render} from 'react-dom';
import resultImg1 from '../images/result1.png';
import resultImg2 from '../images/result2.png';

export default class Results extends React.Component {
  constructor(props) { super(props); }
  render() {
    return (
      <div className="uk-section">
        <h2>Results</h2>
        <p>{this.props.results[0]}</p>
        <h3>Motion Planning (MP) Dataset</h3>
        <p>{this.props.results[1]}</p>
        <table className="uk-table uk-table-divider">
          <thead>
            <tr>
              <th></th>
              <th>Opt</th>
              <th>Exp</th>
              <th>Hmean</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>BF<br/>WA*</td>
              <td>65.8 (63.8, 68.0)<br/>68.4 (66.5, 70.4)</td>
              <td>44.1 (42.8, 45.5)<br/>35.8 (34.5, 37.1)</td>
              <td>44.8 (43.4, 46.3)<br/>40.4 (39.0, 41.8)</td>
            </tr>
            <tr>
              <td>SAIL<br/>SAIL-SL<br/>BB-A*</td>
              <td>5.7 (4.6, 6.8)<br/>3.1 (2.3, 3.8)<br/>31.2 (28.8, 33.5)</td>
              <td>58.0 (56.1, 60.0)<br/>57.6 (55.7, 59.6)<br/>52.0 (50.2, 53.9)</td>
              <td>7.7 (6.4, 9.0)<br/>4.4 (3.5, 5.3)<br/>31.1 (29.2, 33.0)</td>
            </tr>
            <tr>
              <td>Neural BF<br/><b>Neural A*</b></td>
              <td>75.5 (73.8, 77.1)<br/><b>87.7 (86.6, 88.9)</b></td>
              <td>45.9 (44.6, 47.2)<br/>40.1 (38.9, 41.3)</td>
              <td>52.0 (50.7, 53.4)<br/>52.0 (50.7, 53.3)</td>
            </tr>
          </tbody>
        </table>
        <h3>Selected Path Planning Results</h3>
        <p>{this.props.results[2]}</p>
        <img src={resultImg1} className="uk-align-center" alt="" uk-img />
        <h3>Path Planning Results on SSD Dataset</h3>
        <p>{this.props.results[3]}</p>
        <img src={resultImg2} className="uk-align-center" alt="" uk-img />
      </div>
    );
  }
}
