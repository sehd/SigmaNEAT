using Microsoft.Extensions.Logging;
using Orleans;
using SigmaNEAT.ActorCore;
using System.Threading.Tasks;

namespace SigmaNEAT.Actors
{
    public class BasicNetworkGrain : Grain, INetworkGrain
    {
        private readonly ILogger logger;

        public BasicNetworkGrain(ILogger<BasicNetworkGrain> logger)
        {
            this.logger = logger;
        }

        public Task<double> GetOutput(double[] input)
        {
            logger.LogDebug("Getting output...");
            return Task.FromResult(0.0);
        }
    }
}
