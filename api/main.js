const { ApolloServer, gql } = require('apollo-server');
const { PythonShell } = require('python-shell')
const spawnSync = require("child_process").spawnSync;

const typeDefs = gql`

  type AnalysisResults {
    aResult: String
  }

  type Query {
    analyzeReview(review: String!): AnalysisResults
  }
`
const resolvers = {
  Query: {

      analyzeReview: (_, { review },{ res },b,c) => {

        const pythonProcess = spawnSync('python', ['../analyzer/review-analyzer.py', "wassup"]);
        let finalResult = pythonProcess.stdout.toString();
        finalResult = JSON.parse(finalResult);
        return { aResult: finalResult.results }
      }

    }
}

const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({res}) => {return {res}}
 });

server.listen()
