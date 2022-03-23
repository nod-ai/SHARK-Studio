#include "dshark_common.h"
#include "dshark_message.h"
#include "dshark_client.h"
#include "dshark_server.h"
#include "run_module.h"
#include <fstream>

enum class dSHARKMessageType : uint32_t
{
	EvaluateBinary,
	ServerAccept,
	ServerDeny,
	ServerPing,
	MessageAll,
	ServerMessage,
};


class CustomServer : public dshark::server_interface<dSHARKMessageType>
{
public:
	CustomServer(uint16_t nPort) : dshark::server_interface<dSHARKMessageType>(nPort)
	{

	}

protected:
	virtual bool OnClientConnect(std::shared_ptr<dshark::connection<dSHARKMessageType>> client)
	{
		dshark::message<dSHARKMessageType> msg;
		msg.header.id = dSHARKMessageType::ServerAccept;
		client->Send(msg);
		return true;
	}

	// Called when a client appears to have disconnected
	virtual void OnClientDisconnect(std::shared_ptr<dshark::connection<dSHARKMessageType>> client)
	{
		std::cout << "Removing client [" << client->GetID() << "]\n";
	}

	// Called when a message arrives
	virtual void OnMessage(std::shared_ptr<dshark::connection<dSHARKMessageType>> client, dshark::message<dSHARKMessageType>& msg)
	{
		switch (msg.header.id)
		{
		case dSHARKMessageType::EvaluateBinary:
		{
			std::cout << "[" << client->GetID() << "]: evaluate binary\n";
			auto myfile = std::fstream("output/file.vmfb", std::ios::out | std::ios::binary);
			int n = msg.body.size();
			std::cout << n << "\n";
			myfile.write((char*)&msg.body[0], n);
			myfile.close();

			char* f_ = "output/file.vmfb";
			run_module(f_, 0);


			client->Send(msg);
		}
		break;

		case dSHARKMessageType::ServerPing:
		{
			std::cout << "[" << client->GetID() << "]: Server Ping\n";

			// Simply bounce message back to client
			client->Send(msg);
		}
		break;

		case dSHARKMessageType::MessageAll:
		{
			std::cout << "[" << client->GetID() << "]: Message All\n";

			// Construct a new message and send it to all clients
			dshark::message<dSHARKMessageType> msg;
			msg.header.id = dSHARKMessageType::ServerMessage;
			msg << client->GetID();
			MessageAllClients(msg, client);

		}
		break;
		}
	}
};


int main()
{
	CustomServer server(60000);
	server.Start();

	while (1)
	{
		server.Update(-1, true);
	}



	return 0;
}
