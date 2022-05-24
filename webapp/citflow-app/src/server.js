const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mysql      = require('mysql');

const app = express();
const port = 1337;
app.use(cors());

const connection = mysql.createConnection({
  host     : 'localhost',
  user     : 'citflow',
  password : 'citflow',
  database : 'cameradb'
});

app.get('/', function (req, res) {
    connection.connect();

    connection.query('SELECT * FROM Camerafloors LIMIT 1', function (error, results, fields) {
      if (error) throw error;
      res.send(results)
    });

    connection.end();
});
app.listen(port, () => {
 console.log('listening on port: ${port}');
});