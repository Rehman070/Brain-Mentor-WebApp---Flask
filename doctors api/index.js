const jsonServer = require("json-server"); // importing json-server library
const server = jsonServer.create();
const router = jsonServer.router("db.json");
const middlewares = jsonServer.defaults();
const port = process.env.PORT || 8080; //  chose port from here like 8080, 3001

server.use(middlewares);
server.use(router);

server.listen(port);


// {
//     "id": 13,
//     "name": "Prof. Dr. Adil Aziz",
//     "location": "Heart International Hospital, Rawalpindi, Pakistan",
//     "profession": "Spinal Surgeon • Neurologist",
//     "qualifications": "MBBS, FCPS, MCPS, FICS, AO, SPINE",
//     "image": "https://media.licdn.com/dms/image/C5603AQFhHV1HEhy9Ug/profile-displayphoto-shrink_800_800/0/1517525607734?e=2147483647&v=beta&t=T_SuhZOLRxhnICqkqSz0cZ8u0v8IoWKy_-wb12x0QJg"
// }   