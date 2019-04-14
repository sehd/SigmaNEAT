using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Configuration;
using SigmaNEAT.ActorCore;
using System;
using System.Threading.Tasks;

namespace SigmaNEAT.Controller
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Press any key to start controller");
            Console.ReadKey();
            Run().Wait();
        }

        private static async Task Run()
        {
            try
            {
                using (var client = await ConnectClient())
                {
                    await DoClientWork(client);
                    Console.ReadKey();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"\nException while trying to run client: {e.Message}");
                Console.WriteLine("\nPress any key to exit.");
                Console.ReadKey();
            }
        }

        private static async Task<IClusterClient> ConnectClient()
        {
            IClusterClient client;
            client = new ClientBuilder()
                .UseLocalhostClustering()
                .Configure<ClusterOptions>(options =>
                {
                    options.ClusterId = "dev";
                    options.ServiceId = "Server1";
                })
                .ConfigureLogging(logging => logging.AddConsole())
                .Build();

            await client.Connect();
            Console.WriteLine("Client successfully connected to silo host \n");
            return client;
        }

        private static async Task DoClientWork(IClusterClient client)
        {
            var network = client.GetGrain<INetworkGrain>(0);
            var response = await network.GetOutput(new[] { 1.0 });
            Console.WriteLine("\n\n{0}\n\n", response);
        }
    }
}
